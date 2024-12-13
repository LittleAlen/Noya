#pragma once

#include <vector>
#include <math.h>

namespace mycryptonets
{
    struct Params
    {
        vector<double> conv1Weights;
        vector<double> conv1Biases;
        vector<double> conv2Weights;
        vector<double> conv2Biases;
        vector<double> FCWeights;
        vector<double> FCBiases;
    };

    void readInput(size_t batchSize,
                   double normalizationFactor,
                   vector<vector<vector<double>>> &data, //Batch(image1,image2)
                   vector<vector<size_t>> &labels,string filepath)    // Batch(1,3,5,7,8)
    {
        size_t numRows = 3*32*32;
        size_t batch = (size_t)ceil(10000.0 / batchSize);

        vector<double> pixelBatch(numRows, 0.0);
        vector<vector<double>> imageBatch(batchSize, pixelBatch);
        data = vector<vector<vector<double>>>(batch, imageBatch);
        labels = vector<vector<size_t>>(batch, vector<size_t>(batchSize, 0));

        ifstream infile(filepath);
        assert(infile.is_open());

        string line;
        size_t index = 0;
        while (getline(infile, line))
        {
            vector<int> pixelValuesPerImage = extractIntegers(line);
            labels[index / batchSize][index % batchSize] = pixelValuesPerImage[0];
            assert(pixelValuesPerImage.size()==numRows+1);
            for (size_t i = 1; i<pixelValuesPerImage.size(); i ++) 
            {
                data[index / batchSize][index % batchSize][i-1] = pixelValuesPerImage[i] * normalizationFactor;
            }
            index++;
        }

        infile.close();
    }

    Params readParams(string filepath)
    {
        ifstream infile(filepath);
        assert(infile.is_open());

        Params params;
        string line;

        getline(infile, line);
        params.conv1Weights = split(line, ',');
        //assert(params.conv1Weights.size()==3*4*4*25);

        getline(infile, line);
        params.conv1Biases = split(line, ',');
         //assert(params.conv1Biases.size()==25);

        getline(infile, line);
        params.conv2Weights = split(line, ',');
         //assert(params.conv2Weights.size()==25*4*4*50);

        getline(infile, line);
        params.conv2Biases = split(line, ',');
        //assert(params.conv2Biases.size()==50);

        getline(infile, line);
        params.FCWeights = split(line, ',');
         //assert(params.FCWeights.size()==50*7*7*10);

        getline(infile, line);
        params.FCBiases = split(line, ',');
        // assert(params.FCBiases.size()==10);

        infile.close();

        return params;
    }
}; // namespace mycryptonets