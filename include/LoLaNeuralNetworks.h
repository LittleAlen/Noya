#pragma once

#include <math.h>

#include "SealBfvVector.h"

namespace mycryptonets
{
     void fc(vector<SealBfvCiphertext const *> inputPtr,
            vector<SealBfvPlaintext> &weights,
            vector<SealBfvPlaintext> &biases,
            size_t dotLen,
            vector<SealBfvCiphertext> &destination,
            const SealBfvEnvironment &env)
    {
        assert((inputPtr.size() * weights.size()) % (dotLen * dotLen) == 0);
        destination = vector<SealBfvCiphertext>(
            (inputPtr.size() * weights.size()) / (dotLen * dotLen),
            SealBfvCiphertext(biases[0].scale));
#pragma omp parallel for
        for (size_t i = 0; i < weights.size(); i += dotLen)//start
        {
#pragma omp parallel for
            for (size_t j = 0; j < inputPtr.size(); j += dotLen)
            {
                vector<SealBfvCiphertext> dots;
#pragma omp parallel for                
                for (size_t x = 0; x < dotLen; x++)
                {
                    SealBfvCiphertext temp(biases[0].scale);
                    if (weights[i + x].is_zero() || inputPtr[j + x] == nullptr)
                    {
                        continue;
                    }
                    multiply_plain(*inputPtr[j + x], weights[i + x], temp, env);
                    dots.emplace_back(move(temp));
                }
                size_t idx = (i / dotLen) * inputPtr.size() / dotLen + (j / dotLen);
                if (dots.size() > 0)
                {
                    add_many(dots, destination[idx], env);
                }
               
                add_plain_inplace(destination[idx], biases[i / dotLen], env);
            }
        }
    }

    vector<vector<double>> ConvertToConvolutionRepresentation(vector<vector<double>>&data) {
        vector<vector<double>> msgs(25);
        // output is 13*13
        for (int i = 0; i < 13; i++) {
            for (int j = 0; j < 13; j++) {
                // location of left corner
                int x = i * 2;
                int y = j * 2;
                for (int ii = 0; ii < 5; ii++) {
                    for (int jj = 0; jj < 5; jj++) {
                        int xx = x + ii;
                        int yy = y + jj;
                        msgs[ii * 5 + jj].push_back(data[xx][yy]);
                    }
                }
            }
        }
        return msgs;
    }
   
    vector<vector<double>> ConvertToConvolutionRepresentation(vector<vector<vector<vector<double>>>>& data) {
        vector<vector<double>> msgs(3*8*8);
        // output is 14x14
        for(int k=0;k<data.size();k++){
            for (int i = 0; i < 14; i++) {
                for (int j = 0; j < 14; j++) {
                    // location of left corner
                    int x = i * 2;
                    int y = j * 2;
                    for(int d=0;d<3;d++)
                        for (int ii = 0; ii < 8; ii++) {
                            for (int jj = 0; jj < 8; jj++) {
                                int xx = x + ii;
                                int yy = y + jj;
                                msgs[d*8*8+ii * 8 + jj].push_back(data[k][d][xx][yy]);
                            }
                        }
                }
            }
            //padding 256-196=60
            for(int pad=0;pad<60;pad++){
                for(int d=0;d<3;d++)
                    for (int ii = 0; ii < 8; ii++) {
                        for (int jj = 0; jj < 8; jj++) {
                            msgs[d*8*8+ii * 8 + jj].push_back(0);
                        }
                    }
            }

        }
        return msgs;
    }
} // namespace mycryptonets