#pragma once

#include <vector>
#include <math.h>

namespace mycryptonets
{
    struct Params
    {
        vector<double> convWeights;
        vector<double> convBiases;
        vector<double> FC1Weights;
        vector<double> FC1Biases;
        vector<double> FC2Weights;
        vector<double> FC2Biases;
    };

    void readInput(size_t batchSize,
                   double normalizationFactor,
                   vector<vector<vector<double>>> &data, //Batch(image1,image2)
                   vector<vector<size_t>> &labels,string filepath)    // Batch(1,3,5,7,8)
    {
        size_t numRows = 28 * 28;
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

            for (size_t i = 2; i < pixelValuesPerImage.size(); i += 2)
            {
                data[index / batchSize][index % batchSize][pixelValuesPerImage[i]] = pixelValuesPerImage[i + 1] * normalizationFactor;
            }
            index++;
        }

        infile.close();
    }

    void readInputCryptonets(size_t batchSize,
                   double normalizationFactor,
                   vector<vector<vector<double>>> &data,
                   vector<vector<size_t>> &labels,string filepath)
    {
        size_t numRows = 28 * 28;
        size_t batch = (size_t)ceil(10000.0 / batchSize);

        vector<double> pixelBatch(batchSize, 0.0);
        vector<vector<double>> imageBatch(numRows, pixelBatch);
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

            for (size_t i = 2; i < pixelValuesPerImage.size(); i += 2)
            {
                data[index / batchSize][pixelValuesPerImage[i]][index % batchSize] = pixelValuesPerImage[i + 1] * normalizationFactor;
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
        params.convWeights = split(line, ' ');
        assert(params.convWeights.size()==5*5*5);

        getline(infile, line);
        params.convBiases = split(line, ' ');
        assert(params.convBiases.size()==5);

        getline(infile, line);
        params.FC1Weights = split(line, ' ');
        assert(params.FC1Weights.size()==845*100);

        getline(infile, line);
        params.FC1Biases = split(line, ' ');
        assert(params.FC1Biases.size()==100);

        getline(infile, line);
        params.FC2Weights = split(line, ' ');
        assert(params.FC2Weights.size()==100*10);

        getline(infile, line);
        params.FC2Biases = split(line, ' ');
        assert(params.FC2Biases.size()==10);

        infile.close();

        return params;
    }
}; // namespace mycryptonets