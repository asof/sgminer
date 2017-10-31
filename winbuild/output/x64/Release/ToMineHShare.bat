setx GPU_MAX_ALLOC_PERCENT 100
setx GPU_USE_SYNC_OBJECTS 1
setx GPU_USE_OBJECTS  1
setx GPU_MAX_HEAP_SIZE 100


::====== Modify "HD2K9zu8drJ6kE9g39TYA5iCBkPhCXH7Ci" to your own local wallet address of HShare (HSR) ======
sgminer.exe -k x13 -o stratum+tcp://hcash.uupool.cn:51001 -u HD2K9zu8drJ6kE9g39TYA5iCBkPhCXH7Ci.worker1 -p x --intensity 19 --worksize 64 --keccak-unroll 0 --hamsi-expand-big 4

