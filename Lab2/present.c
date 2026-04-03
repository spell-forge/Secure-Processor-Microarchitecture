#include <stdint.h>
#include <string.h>

static const uint8_t SBOX[16] = {
    0xC,0x5,0x6,0xB,0x9,0x0,0xA,0xD,0x3,0xE,0xF,0x8,0x4,0x7,0x1,0x2
};
static const uint8_t PERM[64] = {
     0,16,32,48, 1,17,33,49, 2,18,34,50, 3,19,35,51,
     4,20,36,52, 5,21,37,53, 6,22,38,54, 7,23,39,55,
     8,24,40,56, 9,25,41,57,10,26,42,58,11,27,43,59,
    12,28,44,60,13,29,45,61,14,30,46,62,15,31,47,63
};

static uint64_t do_permute(uint64_t s) {
    uint64_t r = 0;
    for (int i = 0; i < 64; i++) r |= ((s >> (63-i)) & 1ULL) << (63-PERM[i]);
    return r;
}

static void gen_subkeys(const uint8_t *kb, uint64_t *sk) {
    uint64_t kh = 0, kl = 0;
    for (int i=0;i<8;i++) kh=(kh<<8)|kb[i];
    for (int i=8;i<10;i++) kl=(kl<<8)|kb[i];
    sk[0]=kh;
    for (int i=1;i<32;i++) {
        uint64_t t1=kh,t2=kl;
        kh=((t1<<61)|(t2<<45)|(t1>>19))&0xFFFFFFFFFFFFFFFFULL;
        kl=(t1>>3)&0xFFFF;
        uint8_t top=SBOX[(kh>>60)&0xF];
        kh=(kh&0x0FFFFFFFFFFFFFFFULL)|((uint64_t)top<<60);
        kl^=(i&1)<<15; kh^=(i>>1); sk[i]=kh;
    }
}

static int encrypt_check(const uint8_t *pt, const uint8_t *kb, const uint8_t *ct) {
    uint64_t sk[32]; gen_subkeys(kb,sk);
    uint64_t s=0;
    for(int i=0;i<8;i++) s=(s<<8)|pt[i];
    for(int r=0;r<31;r++){
        s^=sk[r]; uint64_t ns=0;
        for(int j=0;j<16;j++){int sh=60-j*4;ns|=(uint64_t)SBOX[(s>>sh)&0xF]<<sh;}
        s=do_permute(ns);
    }
    s^=sk[31];
    uint8_t out[8];
    for(int i=7;i>=0;i--){out[i]=s&0xFF;s>>=8;}
    return memcmp(out,ct,8)==0;
}

int bruteforce_keylow(const uint8_t *rk8, const uint8_t *pt, const uint8_t *ct) {
    uint8_t key[10]; memcpy(key,rk8,8);
    for(int kl=0;kl<65536;kl++){
        key[8]=(kl>>8)&0xFF; key[9]=kl&0xFF;
        if(encrypt_check(pt,key,ct)) return kl;
    }
    return -1;
}
