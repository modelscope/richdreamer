#include <iostream>
#include <cstdint>

extern "C" {
    void zbuff_check(int32_t * src_x, int32_t * src_y, float* depth, int data_size, bool* valid_mask, float * buffs, float * zbuffs, int height, int width) {

        // std::cout<<width<<std::endl;
        for (int i = 0; i < data_size; i++) {
            if (0 == valid_mask[i] ) {
                continue;
            }
            int x = src_x[i];
            int y = src_y[i];
            float z = depth[i];
            if (-1 == buffs[y*width+x] ) {
                buffs[y*width+x] = i;
                zbuffs[y*width+x] = z;
            }
            else{

                if (zbuffs[y*width+x] > z){
                    buffs[y*width+x] = i;
                    zbuffs[y*width+x] = z;
                }
            }
        }
    }
}
