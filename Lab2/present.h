#ifndef PRESENT_H
#define PRESENT_H

#include <stdint.h>

void present_encrypt(uint8_t *block, const uint8_t *key);
void present_decrypt(uint8_t *block, const uint8_t *key);

#endif
