#include "LoLaNeuralNetworks.h"
#include "MnistDataPreprocess.h"
#include "SRHE.h"
using namespace std;
using namespace seal;
using namespace mycryptonets;

namespace {
    size_t POLY_MODULUS_DEGREE = 8192; 
    
    // vector<Modulus> PLAINTEXT_MODULUS = PlainModulus::Batching(POLY_MODULUS_DEGREE, {18,18});
    // double INPUT_SCALE = 8*2;
    // double CONV_SCALE = 32;//32*4
    // double FC1_SCALE = 32*2; //32*32
    // double FC2_SCALE = 32*2; //32*32

    // vector<Modulus> PLAINTEXT_MODULUS ={557057, 638977,737281, 786433};
    // double INPUT_SCALE = 16;
    // double CONV_SCALE = 32;
    // double FC1_SCALE = 32*32;
    // double FC2_SCALE = 32;

    vector<Modulus> PLAINTEXT_MODULUS = PlainModulus::Batching(POLY_MODULUS_DEGREE, {18,18,20});
    double INPUT_SCALE = 8;
    double CONV_SCALE = 32;
    double FC1_SCALE = 32*2;
    double FC2_SCALE = 32*2;
}


vector<SealBfvCiphertext> LoLa(const Params &params, const vector<SealBfvCiphertext> &inputE, const SealBfvEnvironment &env)
{
    vector<SealBfvCiphertext> intermediateResultsE;

    std::chrono::time_point<std::chrono::steady_clock> start, end;
    std::chrono::duration<double> elapsed_seconds;
    // Conv Layer
    {
        vector<SealBfvCiphertext> resultE;
        auto decor = make_decorator(fc, "Conv");

        vector<SealBfvPlaintext> WeightsP;
        vector<SealBfvPlaintext> BiasesP;

        encode(params.convWeights, WeightsP, env, CONV_SCALE);
        encode(params.convBiases, BiasesP, env, inputE[0].scale * CONV_SCALE,169);
        decor(getPointers(inputE),
              WeightsP,
              BiasesP,
              5 * 5,
              resultE,
              env);
        intermediateResultsE = move(resultE);

        cout << "- Noise budget: " << env.environments[0].decryptorPtr->invariant_noise_budget(intermediateResultsE[0].eVectors[0]) << endl;
    }
    //Combining 5 ciphertext to one  845X1
    {
        vector<SealBfvCiphertext> all(1,SealBfvCiphertext(intermediateResultsE[0].scale));
        for(int i=1;i<intermediateResultsE.size();i++){
            rotate_inplace(intermediateResultsE[i],-169*i,env);
        }

        SealBfvCiphertext & resultE=all[0];
        add_many(intermediateResultsE,resultE,env);
        intermediateResultsE.clear();
        intermediateResultsE = move(all);
    }
    //stacking 845
    {
        vector<SealBfvCiphertext> resultE;
        rotate_add(intermediateResultsE[0],vector<int>({1024*-1,1024*-2}),intermediateResultsE[0],env);
        SealBfvCiphertext tmp(intermediateResultsE[0]);
        rotate_columns_inplace(tmp,env);
        intermediateResultsE.emplace_back(move(tmp));
        add_many(intermediateResultsE,tmp,env);
        intermediateResultsE.clear();
        intermediateResultsE.emplace_back(move(tmp));

    }


    // Square activation layer
    {
        auto decor = make_decorator(square_inplace_vec, "Square");

        decor(intermediateResultsE, env);

        cout << "- Noise budget: " << env.environments[0].decryptorPtr->invariant_noise_budget(intermediateResultsE[0].eVectors[0]) << endl;
    }
    
    // replace to SRHE layer(Simulating Relu by Homomorphic Encrytion)
    //simulate_relu(intermediateResultsE,env);

    // Modulus switch
    {
        #pragma omp parallel for
        for (size_t i = 0; i < intermediateResultsE.size(); i++) {
            mod_switch_to_next_inplace(intermediateResultsE[i], env);
        }
    }

    // FC1 Layer
    {
        std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();

        vector<SealBfvCiphertext> resultE;

        vector<SealBfvPlaintext> WeightsP;
        SealBfvPlaintext BiasesP;
        // 100 = 12*8 + 4
        for (int i = 0; i < 13; i++) {
            vector<double> tmp(env.poly_modulus_degree, 0.0);
            auto rowStartAddr = &params.FC1Weights[i * 8 * 845];
            int last = i == 12 ? 4 : 8;
            for (int j = 0; j < last; j++) {
                std::copy(rowStartAddr + j * 845, rowStartAddr + (j + 1) * 845, tmp.begin() + j * 1024);
            }
            WeightsP.emplace_back(move(SealBfvPlaintext(tmp,env,FC1_SCALE)));
        }
        
        vector<double> vMsgs(env.poly_modulus_degree,0.0);
        for (int i = 1; i <= 8; i++) {
            vMsgs[i * 1024 - 1] = 1.0;
        }
        SealBfvPlaintext slotMask(vMsgs,env,1);
       

        std::fill(vMsgs.begin(), vMsgs.end(), 0.0);
        // 100 = 4 * 13 + 4 * 12
        for (int i = 0; i < 8; i++) {
            int last = 13;
            if (i > 3) last = 12;
            for (int j = 0; j < last; j++) {
                int index = ((i + 1) * 1024 - 1 -j) % env.poly_modulus_degree;
                int target = i + 8 * j;
                vMsgs[index] = params.FC1Biases[target];
            }
        }
        BiasesP=SealBfvPlaintext(vMsgs,env,intermediateResultsE[0].scale * FC1_SCALE);
        
        SealBfvCiphertext & ciphertext= intermediateResultsE[0];
        vector<SealBfvCiphertext> all(13,SealBfvCiphertext(ciphertext.scale));
        SealBfvCiphertext tmp(WeightsP[0].scale);
        for(int i=0;i<WeightsP.size();i++){
            multiply_plain(ciphertext,WeightsP[i],tmp,env);
            rotate_add(tmp,vector<int>({-1,-2,-4,-8,-16,-32,-64,-128,-256,-512}),all[i],env);
            multiply_plain(all[i],slotMask,tmp,env);
            if(i!=0)
                rotate_inplace(tmp,i,env);
            all[i]=tmp;
        }
        add_many(all,tmp,env);
        add_plain_inplace(tmp,BiasesP,env);
        resultE.emplace_back(move(tmp));
        intermediateResultsE = move(resultE);
        std::chrono::time_point<std::chrono::steady_clock> end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        cout<<"FC1 compute time: "<<elapsed_seconds.count()<<endl;
        cout << "- Noise budget: " << env.environments[0].decryptorPtr->invariant_noise_budget(intermediateResultsE[0].eVectors[0]) << endl;
        
    }//100X1

    //Modulus switch
    // {
    //     #pragma omp parallel for
    //     for (size_t i = 0; i < intermediateResultsE.size(); i++) {
    //         mod_switch_to_next_inplace(intermediateResultsE[i], env);
    //     }
    // }
    // Square activation layer
    {
        auto decor = make_decorator(square_inplace_vec, "Square");

        decor(intermediateResultsE, env);

        cout << "- Noise budget: " << env.environments[0].decryptorPtr->invariant_noise_budget(intermediateResultsE[0].eVectors[0]) << endl;
    }
    // replace to SRHE layer(Simulating Relu by Homomorphic Encrytion)
    //simulate_relu(intermediateResultsE,env);
    
    // Modulus switch
    {
        #pragma omp parallel for
        for (size_t i = 0; i < intermediateResultsE.size(); i++) {
            mod_switch_to_next_inplace(intermediateResultsE[i], env);
        }
    }
    //Debug Only
    // vector<vector<double>> res;
    // decrypt(intermediateResultsE,res,env);
    // encrypt(res,intermediateResultsE,env,intermediateResultsE[0].scale);


    // FC2 Layer
    {
        std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
        vector<SealBfvCiphertext> resultE(10,SealBfvCiphertext(intermediateResultsE[0].scale));
        auto decor = make_decorator(fc, "FC2");
        vector<SealBfvPlaintext> WeightsP;
        vector<SealBfvPlaintext> BiasesP;
        vector<double> vMsgs(env.poly_modulus_degree,0.0);
        for (int k = 0; k < 10; k++) {
            std::fill(vMsgs.begin(), vMsgs.end(), 0.0);
            for (int i = 0; i < 8; i++) { //8 start points
                int last = 13;
                if (i > 3) last = 12;
                for (int j = 0; j < last; j++) {
                    int index = ((i + 1) * 1024 - 1 -j) % env.poly_modulus_degree;
                    int target = i + 8 * j;
                    vMsgs[index] = params.FC2Weights[k * 100 + target];
                }
            }
            WeightsP.emplace_back(move(SealBfvPlaintext(vMsgs,env,FC2_SCALE)));
        }
        size_t Bias_Scale=intermediateResultsE[0].scale * FC2_SCALE;
        std::fill(vMsgs.begin(), vMsgs.end(), 0.0);
        for(auto bias:params.FC2Biases){
            vMsgs[8176]=bias;
            BiasesP.emplace_back(move(SealBfvPlaintext(vMsgs,env,Bias_Scale)));
        }
        SealBfvCiphertext tmp(intermediateResultsE[0].scale);
        SealBfvCiphertext& ciphertext=intermediateResultsE[0];

        std::fill(vMsgs.begin(), vMsgs.end(), 0.0);
        vMsgs[4080]=1;
        vMsgs[8176]=1;
        SealBfvPlaintext slotMask(vMsgs,env,1);
        for(int i=0;i<WeightsP.size();i++){
            multiply_plain(ciphertext,WeightsP[i],tmp,env);
            rotate_add(tmp,vector<int>({1,2,4,8,-1024,-1024*2}),resultE[i],env);// note that the result is the sum of 4095th and 8191th
            // multiply_plain(resultE[i],slotMask,resultE[i],env);
            add_plain_inplace(resultE[i],BiasesP[i],env);
        }
        //1023-15=1008
        //1008+1024*3=4080   8176
        intermediateResultsE.clear();
        intermediateResultsE = move(resultE);
        std::chrono::time_point<std::chrono::steady_clock> end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;

        cout<<"FC2 compute time: "<<elapsed_seconds.count()<<endl;
       
        // Modulus switch
        {
            #pragma omp parallel for
            for (size_t i = 0; i < intermediateResultsE.size(); i++) {
                mod_switch_to_next_inplace(intermediateResultsE[i], env);
            }
        }

        cout << "- Noise budget: " << env.environments[0].decryptorPtr->invariant_noise_budget(intermediateResultsE[0].eVectors[0]) << endl;
        
    }
    
    // vector<vector<double>> res;
    // decrypt(intermediateResultsE, res, env);

    return intermediateResultsE;
}
void Test(SealBfvEnvironment &env){
    vector<vector<double>>v(1,vector<double>(POLY_MODULUS_DEGREE,1000));
    vector<SealBfvCiphertext> c;
    SealBfvPlaintext p(1000,env,1);
    encrypt(v,c,env,1);
    cout << " Start Noise budget: " << env.environments[0].decryptorPtr->invariant_noise_budget(c[0].eVectors[0]) << endl;
    SealBfvCiphertext tmp=c[0];
    square_inplace(tmp,env);
    cout << " Square Noise budget: " << env.environments[0].decryptorPtr->invariant_noise_budget(tmp.eVectors[0]) << endl;
    tmp=c[0];
    multiply_plain(tmp,p,tmp,env);
    cout << " Multiplain  Noise budget: " << env.environments[0].decryptorPtr->invariant_noise_budget(tmp.eVectors[0]) << endl;
}
int main()
{
    SealBfvEnvironment env = SealBfvEnvironment(
        POLY_MODULUS_DEGREE,
        PLAINTEXT_MODULUS);

    // Test(env);
    auto params = readParams("./resources/MnistSquare9665.txt");

    vector<vector<vector<double>>> data;
    vector<vector<size_t>> labels;
    // readInput(1, 1.0, data, labels,"./resources/MNIST-28x28-test.txt");
    readInput(1, 1.0/255, data, labels,"./resources/MNIST-28x28-test.txt");

    // Batch processing

    size_t correct = 0;
    for (size_t batchIndex = 0; batchIndex < data.size(); batchIndex++)
    {
        //Conv Organization 
        vector<vector<double>> image_data(29, vector<double>(29, 0.0));
        // zero-pad image
        for (int i = 1; i <= 28; i++) {
            for (int j = 1; j <= 28; j++) {
                image_data[i][j] = data[batchIndex][0][(i - 1) * 28 + (j - 1)];
            }
        }
        vector<vector<double>> msgs = ConvertToConvolutionRepresentation(image_data);
        assert(msgs[0].size()==169);

        // Encrypt
        vector<SealBfvCiphertext> inputE;
        auto decor = make_decorator(encrypt, "Encryption");
        decor(msgs, inputE, env, INPUT_SCALE);
        cout << "- Noise budget: " << env.environments[0].decryptorPtr->invariant_noise_budget(inputE[0].eVectors[0]) << endl;

        // Forward pass on the cloud
        std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
        auto intermediateResultsE = LoLa(params, inputE, env);
        std::chrono::time_point<std::chrono::steady_clock> end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        cout << setw(20) << "Total time: ";
        cout << elapsed_seconds.count() << " seconds" << std::endl;

        //output 10 ciphertext
        //Decrypt(user)
        vector<vector<double>> res;
        auto decorD=make_decorator(decrypt,"Decryption");
        decorD(intermediateResultsE, res, env);//res 10X8192 的double值
        vector<double>val;
        for(int i=0;i<res.size();i++){
            val.push_back(res[i][4080]+res[i][8176]);
        }
        auto predicted = std::distance(val.begin(), std::max_element(val.begin(), val.end()));
        cout << labels[batchIndex][0] << " vs " << predicted << endl;

         correct += labels[batchIndex][0] == predicted ? 1 : 0;


        // break;
    }
     cout << "Accuracy is " << correct << "/" << 10000;
        cout << "=" << (1.0 * correct)/10000 << endl;
    return 0;
}