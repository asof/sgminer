/*
* X13 kernel implementation.
*
* ==========================(LICENSE BEGIN)============================
*
* Copyright (c) 2014  phm
* Copyright (c) 2014 Girino Vey
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to deal in the Software without restriction, including
* without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*
* ===========================(LICENSE END)=============================
*
* @author   phm <phm@inbox.com>
*/

#ifndef X13MOD_CL
#define X13MOD_CL

#define DEBUG(x)

#if __ENDIAN_LITTLE__
#define SPH_LITTLE_ENDIAN 1
#else
#define SPH_BIG_ENDIAN 1
#endif

#define SPH_UPTR sph_u64

typedef unsigned int sph_u32;
typedef int sph_s32;
#ifndef __OPENCL_VERSION__
typedef unsigned long long sph_u64;
typedef long long sph_s64;
#else
typedef unsigned long sph_u64;
typedef long sph_s64;
#endif

#define SPH_64 1
#define SPH_64_TRUE 1

#define SPH_C32(x)    ((sph_u32)(x ## U))
#define SPH_T32(x) (as_uint(x))
#define SPH_ROTL32(x, n) rotate(as_uint(x), as_uint(n))
#define SPH_ROTR32(x, n)   SPH_ROTL32(x, (32 - (n)))

#define SPH_C64(x)    ((sph_u64)(x ## UL))
#define SPH_T64(x) (as_ulong(x))
#define SPH_ROTL64(x, n) rotate(as_ulong(x), (n) & 0xFFFFFFFFFFFFFFFFUL)
#define SPH_ROTR64(x, n)   SPH_ROTL64(x, (64 - (n)))

#define SPH_ECHO_64 1
#define SPH_KECCAK_64 1
#define SPH_JH_64 1
#define SPH_SIMD_NOCOPY 0
#define SPH_KECCAK_NOCOPY 0
#define SPH_SMALL_FOOTPRINT_GROESTL 0
#define SPH_GROESTL_BIG_ENDIAN 0
#define SPH_CUBEHASH_UNROLL 0

#ifndef SPH_COMPACT_BLAKE_64
#define SPH_COMPACT_BLAKE_64 0
#endif
#ifndef SPH_LUFFA_PARALLEL
#define SPH_LUFFA_PARALLEL 0
#endif
#ifndef SPH_KECCAK_UNROLL
#define SPH_KECCAK_UNROLL 0
#endif
#ifndef SPH_HAMSI_EXPAND_BIG
#define SPH_HAMSI_EXPAND_BIG 1
#endif

#define SWAP4(x) as_uint(as_uchar4(x).wzyx)
#define SWAP8(x) as_ulong(as_uchar8(x).s76543210)

#if SPH_BIG_ENDIAN
#define DEC64E(x) (x)
#define DEC64BE(x) (*(const __global sph_u64 *) (x));
#else
#define DEC64E(x) SWAP8(x)
#define DEC64BE(x) SWAP8(*(const __global sph_u64 *) (x));
#endif

#define SHL(x, n) ((x) << (n))
#define SHR(x, n) ((x) >> (n))

#define CONST_EXP2  q[i+0] + SPH_ROTL64(q[i+1], 5)  + q[i+2] + SPH_ROTL64(q[i+3], 11) + \
                    q[i+4] + SPH_ROTL64(q[i+5], 27) + q[i+6] + SPH_ROTL64(q[i+7], 32) + \
                    q[i+8] + SPH_ROTL64(q[i+9], 37) + q[i+10] + SPH_ROTL64(q[i+11], 43) + \
                    q[i+12] + SPH_ROTL64(q[i+13], 53) + (SHR(q[i+14],1) ^ q[i+14]) + (SHR(q[i+15],2) ^ q[i+15])

typedef union {
	unsigned char h1[64];
	uint h4[16];
	ulong h8[8];
} hash_t;




#define cpu_to_be16(v) (((v)<< 8) | ((v)>>8))
#define cpu_to_be32(v) (((v)>>24) | (((v)>>8)&0xff00) | (((v)<<8)&0xff0000) | ((v)<<24))
#define be16_to_cpu(v) cpu_to_be16(v)
#define be32_to_cpu(v) cpu_to_be32(v)


#define SM3_ROTATELEFT(X,n)  (((X)<<(n)) | ((X)>>(32-(n))))
//#define SM3_ROTATELEFT(x, n) rotate(as_uint(x), as_uint(n))
//#define SM3_ROTATELEFT(x, n) rotate((x), (n))
//#define SM3_ROTATELEFT(x, y)		rotate(x, y)

#define SM3_P0(x) ((x) ^  SM3_ROTATELEFT((x),9)  ^ SM3_ROTATELEFT((x),17))
#define SM3_P1(x) ((x) ^  SM3_ROTATELEFT((x),15) ^ SM3_ROTATELEFT((x),23))

#define SM3_FF0(x,y,z) ( (x) ^ (y) ^ (z))
#define SM3_FF1(x,y,z) (((x) & (y)) | ( (x) & (z)) | ( (y) & (z)))

#define SM3_GG0(x,y,z) ( (x) ^ (y) ^ (z))
#define SM3_GG1(x,y,z) (((x) & (y)) | ( (~(x)) & (z)) )


#define SM3_DIGEST_LENGTH	32
#define SM3_BLOCK_SIZE		64
#define SM3_CBLOCK		(SM3_BLOCK_SIZE)
#define SM3_HMAC_SIZE		(SM3_DIGEST_LENGTH)


__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search10b(__global hash_t* hashes)
{
	uint gid = get_global_id(0);
	uint offset = get_global_offset(0);
	__global hash_t *hash = &(hashes[gid - offset]);
	// sm3
	{
		int jj;
		hash_t ctx_block;
		uint digest[8] = { 0 };
		uint W[68] = { 0 };
		uint W1[64] = { 0 };
		uint T[64] = { 0 };
		uint SS1, SS2, TT1, TT2;
		uint A, B, C, D, E, F, G, H;
		SS1 = SS2 = TT1 = TT2 = 0;
		A = B = C = D = E = F = G = H = 0;
		for (jj = 0; jj<64; jj++){
			ctx_block.h1[jj] = 0;
		}
		ctx_block.h1[0] = 0x80;
		ctx_block.h4[14] = cpu_to_be32(1 >> 23);
		ctx_block.h4[15] = cpu_to_be32(1 << 9);
		digest[0] = 0x7380166F;
		digest[1] = 0x4914B2B9;
		digest[2] = 0x172442D7;
		digest[3] = 0xDA8A0600;
		digest[4] = 0xA96F30BC;
		digest[5] = 0x163138AA;
		digest[6] = 0xE38DEE4D;
		digest[7] = 0xB0FB0E4E;
		//sm3_compress(ctx_digest, hash->h4);
		A = digest[0];
		B = digest[1];
		C = digest[2];
		D = digest[3];
		E = digest[4];
		F = digest[5];
		G = digest[6];
		H = digest[7];

		for (jj = 0; jj < 16; jj++) {
			W[jj] = cpu_to_be32(hash->h4[jj]);
		}
		for (jj = 16; jj < 68; jj++) {
			W[jj] = SM3_P1(W[jj - 16] ^ W[jj - 9] ^ SM3_ROTATELEFT(W[jj - 3], 15)) ^ SM3_ROTATELEFT(W[jj - 13], 7) ^ W[jj - 6];;
		}
		for (jj = 0; jj < 64; jj++) {
			W1[jj] = W[jj] ^ W[jj + 4];
		}
		for (jj = 0; jj < 16; jj++) {

			T[jj] = 0x79CC4519;
			SS1 = SM3_ROTATELEFT((SM3_ROTATELEFT(A, 12) + E + SM3_ROTATELEFT(T[jj], jj)), 7);
			SS2 = SS1 ^ SM3_ROTATELEFT(A, 12);
			TT1 = SM3_FF0(A, B, C) + D + SS2 + W1[jj];
			TT2 = SM3_GG0(E, F, G) + H + SS1 + W[jj];
			D = C;
			C = SM3_ROTATELEFT(B, 9);
			B = A;
			A = TT1;
			H = G;
			G = SM3_ROTATELEFT(F, 19);
			F = E;
			E = SM3_P0(TT2);
		}
		for (jj = 16; jj < 64; jj++) {
			T[jj] = 0x7A879D8A;
			SS1 = SM3_ROTATELEFT((SM3_ROTATELEFT(A, 12) + E + SM3_ROTATELEFT(T[jj], jj)), 7);
			SS2 = SS1 ^ SM3_ROTATELEFT(A, 12);
			TT1 = SM3_FF1(A, B, C) + D + SS2 + W1[jj];

			hash->h4[0] = E;
			hash->h4[1] = F;
			hash->h4[2] = G;
			hash->h4[3] = H;
			hash->h4[4] = SS1;
			hash->h4[5] = W[jj];

			//TT2 = SM3_GG1(E,F,G) + H + SS1 + W[jj];
			sph_u32 xg1 = E & F;
			sph_u32 xg2 = ~E;
			sph_u32 xg3 = xg2 & G;
			sph_u32 xg4 = xg1 + xg3;

			TT2 = xg4 + H + SS1 + W[jj];

			hash->h4[6] = TT2;

			hash->h4[7] = xg1;
			hash->h4[8] = xg2;
			hash->h4[9] = xg3;
			hash->h4[10] = xg4;

			D = C;
			C = SM3_ROTATELEFT(B, 9);
			B = A;
			A = TT1;
			H = G;
			G = SM3_ROTATELEFT(F, 19);
			F = E;
			E = SM3_P0(TT2);
		}

		hash->h4[0] = A;
		hash->h4[1] = B;
		hash->h4[2] = C;
		hash->h4[3] = D;
		hash->h4[4] = E;
		hash->h4[5] = F;
		hash->h4[6] = G;
		hash->h4[7] = H;
		hash->h4[8] = SS1;
		hash->h4[9] = SS2;
		hash->h4[10] = TT1;
		hash->h4[11] = TT2;

		digest[0] ^= A;
		digest[1] ^= B;
		digest[2] ^= C;
		digest[3] ^= D;
		digest[4] ^= E;
		digest[5] ^= F;
		digest[6] ^= G;
		digest[7] ^= H;

		for (jj = 0; jj<8; jj++){
			hash->h4[jj] = digest[jj];
		}

		//sm3_compress(ctx_digest, ctx_block);
		A = digest[0];
		B = digest[1];
		C = digest[2];
		D = digest[3];
		E = digest[4];
		F = digest[5];
		G = digest[6];
		H = digest[7];

		for (jj = 0; jj < 16; jj++) {
			W[jj] = cpu_to_be32(ctx_block.h4[jj]);
		}
		for (jj = 16; jj < 68; jj++) {
			W[jj] = SM3_P1(W[jj - 16] ^ W[jj - 9] ^ SM3_ROTATELEFT(W[jj - 3], 15)) ^ SM3_ROTATELEFT(W[jj - 13], 7) ^ W[jj - 6];;
		}
		for (jj = 0; jj < 64; jj++) {
			W1[jj] = W[jj] ^ W[jj + 4];
		}

		for (jj = 0; jj < 16; jj++) {
			T[jj] = 0x79CC4519;
			SS1 = SM3_ROTATELEFT((SM3_ROTATELEFT(A, 12) + E + SM3_ROTATELEFT(T[jj], jj)), 7);
			SS2 = SS1 ^ SM3_ROTATELEFT(A, 12);
			TT1 = SM3_FF0(A, B, C) + D + SS2 + W1[jj];
			TT2 = SM3_GG0(E, F, G) + H + SS1 + W[jj];
			D = C;
			C = SM3_ROTATELEFT(B, 9);
			B = A;
			A = TT1;
			H = G;
			G = SM3_ROTATELEFT(F, 19);
			F = E;
			E = SM3_P0(TT2);
		}

		for (jj = 16; jj < 64; jj++) {

			T[jj] = 0x7A879D8A;
			SS1 = SM3_ROTATELEFT((SM3_ROTATELEFT(A, 12) + E + SM3_ROTATELEFT(T[jj], jj)), 7);
			SS2 = SS1 ^ SM3_ROTATELEFT(A, 12);
			TT1 = SM3_FF1(A, B, C) + D + SS2 + W1[jj];
			//TT2 = SM3_GG1(E,F,G) + H + SS1 + W[jj];

			sph_u32 xg1 = E & F;
			sph_u32 xg2 = ~E;
			sph_u32 xg3 = xg2 & G;
			sph_u32 xg4 = xg1 + xg3;

			TT2 = xg4 + H + SS1 + W[jj];

			D = C;
			C = SM3_ROTATELEFT(B, 9);
			B = A;
			A = TT1;
			H = G;
			G = SM3_ROTATELEFT(F, 19);
			F = E;
			E = SM3_P0(TT2);
		}
		digest[0] ^= A;
		digest[1] ^= B;
		digest[2] ^= C;
		digest[3] ^= D;
		digest[4] ^= E;
		digest[5] ^= F;
		digest[6] ^= G;
		digest[7] ^= H;
		for (jj = 0; jj < 8; jj++) {
			hash->h4[jj] = cpu_to_be32(digest[jj]);
		}
		for (jj = 8; jj < 16; jj++) {
			hash->h4[jj] = 0;
		}
	}
}


#endif // X13MOD_CL

