//the network we use is conv 50 conv 100 fc10
#include "NoyaNeuralNetworks.h"
#include "CifarDataPreprocess.h"
#include "SRHE.h"
using namespace std;
using namespace seal;
using namespace mycryptonets;

namespace {
    size_t POLY_MODULUS_DEGREE = 8192; 
    size_t BATCH_SIZE=32;
    // vector<Modulus> PLAINTEXT_MODULUS = PlainModulus::Batching(
    //     POLY_MODULUS_DEGREE, {18*2, 18*2});
    
    // double INPUT_SCALE = 1;
    // double CONV1_SCALE = 256;
    // double CONV2_SCALE = 512;
    // double FC_SCALE = 512;
    vector<Modulus> PLAINTEXT_MODULUS = PlainModulus::Batching(
    POLY_MODULUS_DEGREE, {18, 18});
    
    double INPUT_SCALE = 8*2;
    double CONV1_SCALE = 32*8;
    double CONV2_SCALE = 32*16;
    double FC_SCALE = 32*32;
    
   
}


vector<SealBfvCiphertext> Noya(const Params &params, const vector<SealBfvCiphertext> &inputE, const SealBfvEnvironment &env)
{
    cout<<"start Noya"<<endl;
    vector<SealBfvCiphertext> intermediateResultsE;
   
    std::chrono::time_point<std::chrono::steady_clock> start, end;
    std::chrono::duration<double> elapsed_seconds;
    // Conv1 Layer
    {
        vector<SealBfvCiphertext> resultE;
        auto decor = make_decorator(fc, "Conv1");

        vector<SealBfvPlaintext> WeightsP;
        vector<SealBfvPlaintext> BiasesP;

        encode(params.conv1Weights, WeightsP, env, CONV1_SCALE);
        encode_padding(params.conv1Biases, BiasesP, env, inputE[0].scale * CONV1_SCALE,256,256);
        decor(getPointers(inputE),
              WeightsP,
              BiasesP,
              3*4*4,
              resultE,
              env);
        intermediateResultsE = move(resultE);
        cout << "- Noise budget: " << env.environments[0].decryptorPtr->invariant_noise_budget(intermediateResultsE[0].eVectors[0]) << endl;
    }
    
    //debug
    // {
    //     vector<vector<double>> res;
    //     decrypt(intermediateResultsE, res, env);
    //     stringstream s;
    //     for(int i=0;i<res[0].size();i++)
    //         s<<res[0][i]<<" ";
    //     s<<endl;
    //     ofstream outfile("./debug-conv1.txt",ios::out);
    //     outfile << s.str();
    //     outfile.close();
    // }
    // Square activation layer
    // {
    //     auto decor = make_decorator(square_inplace_vec, "Square");

    //     decor(intermediateResultsE, env);
    //     cout<<"-Scale :"<<intermediateResultsE[0].scale<<endl;
    //     cout << "- Noise budget: " << env.environments[0].decryptorPtr->invariant_noise_budget(intermediateResultsE[0].eVectors[0]) << endl;
    // }
    
    // replace to SRHE layer(Simulating Relu by Homomorphic Encrytion)
    simulate_relu(intermediateResultsE,env);
    // //debug
    // {
    //     vector<vector<double>> res;
    //     decrypt(intermediateResultsE, res, env);
    //     stringstream s;
    //     for(int i=0;i<res[0].size();i++)
    //         s<<res[0][i]<<" ";
    //     s<<endl;
    //     ofstream outfile("./debug-srhe1.txt",ios::out);
    //     outfile << s.str();
    //     outfile.close();
    // }
    // Modulus switch
    {
        #pragma omp parallel for
        for (size_t i = 0; i < intermediateResultsE.size(); i++) {
            mod_switch_to_next_inplace(intermediateResultsE[i], env);
        }
    }

    //Debug Only , we can use it for unlimited noisy budget
    // vector<vector<double>> res1;
    // decrypt(intermediateResultsE,res1,env);
    // encrypt(res1,intermediateResultsE,env,intermediateResultsE[0].scale);
  
    // Conv2 Layer
    {
        start = std::chrono::steady_clock::now();

        vector<SealBfvCiphertext> resultE;

        vector<vector<vector<double>>> weight(50*7*7,vector<vector<double>>(25,vector<double>(env.poly_modulus_degree,0.0)));
        vector<SealBfvPlaintext>WeightsP(50*7*7*25,SealBfvPlaintext());
        vector<SealBfvPlaintext> BiasesP(50*7*7,SealBfvPlaintext());
        #pragma omp parallel for
        for(int m=0;m<50;m++){
            #pragma omp parallel for
            for(int i=0;i<7;i++){
                for(int j=0;j<7;j++){
                    //compute the start point
                    int x=i*2;
                    int y=j*2;
                    for(int k=0;k<25;k++){
                        for(int ii=0;ii<4;ii++){
                            for(int jj=0;jj<4;jj++){
                                // if(x+ii<14&&y+jj<14) //we need to ignore some points, because the second convolution's padding is 4,which means 22x22
                                // {
                                    for(int b=0;b<BATCH_SIZE;b++){
                                        //cout<<"weight["<<m*7*7+i*7+j<<"]"<<"["<<k<<"]"<<"["<<b*256+(x+ii)*14+y+jj<<"]"<<"    params.conv2weight["<<m*83*10*10+k*10*10+ii*10+jj<<"]"<<endl;
                                         weight[m*7*7+i*7+j][k][b*256+(x+ii)*16+y+jj]=params.conv2Weights[m*25*4*4+k*4*4+ii*4+jj];}
                                // }
                            }
                        }
                    }
                }
            }
        }

        for(int i=0;i<50*7*7;i++){
            for(int j=0;j<25;j++){
                WeightsP[i*25+j]=move(SealBfvPlaintext(weight[i][j],env,CONV2_SCALE));
            }
        }
        weight.clear();
        vector<vector<vector<double>>>().swap(weight);
       
        
        
        for(int i=0;i<50;i++){
            for(int j=0;j<7*7;j++){
                vector<double> vMsgs(env.poly_modulus_degree,0.0);
                for(int k=0;k<BATCH_SIZE;k++){
                    vMsgs[k*256]=params.conv2Biases[i]; 
                }
                BiasesP[i*7*7+j]=move(SealBfvPlaintext(vMsgs,env,intermediateResultsE[0].scale*CONV2_SCALE));
            }
        }
        fc(getPointers(intermediateResultsE),WeightsP,BiasesP,25,resultE,env);
        intermediateResultsE.clear();
        vector<SealBfvCiphertext>().swap(intermediateResultsE);
        vector<double> vMsgs(env.poly_modulus_degree,0.0);
        for(int k=0;k<BATCH_SIZE;k++)
            vMsgs[k*256]=1; 
        SealBfvPlaintext slotMask(vMsgs,env,1);
        
        vector<int>steps={1,2,4,8,16,32,64,128};
        //convert it to 10 ciphertext
        for(int i=0;i<10;i++){
            vector<SealBfvCiphertext> all;
            SealBfvCiphertext tmp;
            // #pragma omp parallel for
            for(int j=0;j<256;j++){
                if(i*256+j<resultE.size()){
                    rotate_add(resultE[i*256+j],steps,tmp,env);
                    multiply_plain(tmp,slotMask,tmp,env);
                    rotate_inplace(tmp,-1*j,env);
                    all.emplace_back(move(tmp));
                }
            }
            add_many(all,tmp,env);
            intermediateResultsE.emplace_back(move(tmp));
        }
        end = std::chrono::steady_clock::now();
        elapsed_seconds = end - start;
        cout<<"Conv2 compute time: "<<elapsed_seconds.count()<<endl;
        cout << "- Noise budget: " << env.environments[0].decryptorPtr->invariant_noise_budget(intermediateResultsE[0].eVectors[0]) << endl;
        
    }

    //debug
    // {
    //     vector<vector<double>> res;
    //     decrypt(intermediateResultsE, res, env);
    //     stringstream s;
    //     for(int i=0;i<res[0].size();i++)
    //         s<<res[0][i]<<" ";
    //     s<<endl;
    //     ofstream outfile("./debug-conv2.txt",ios::out);
    //     outfile << s.str();
    //     outfile.close();
    // }

    // Modulus switch
    {
        #pragma omp parallel for
        for (size_t i = 0; i < intermediateResultsE.size(); i++) {
            mod_switch_to_next_inplace(intermediateResultsE[i], env);
        }
    }
    // Square activation layer
    // {
    //     auto decor = make_decorator(square_inplace_vec, "Square");

    //     decor(intermediateResultsE, env);

    //     cout << "- Noise budget: " << env.environments[0].decryptorPtr->invariant_noise_budget(intermediateResultsE[0].eVectors[0]) << endl;
    // }
    //replace to SRHE layer(Simulating Relu by Homomorphic Encrytion)
    simulate_relu(intermediateResultsE,env);
    
    //debug
    // {
    //     vector<vector<double>> res;
    //     decrypt(intermediateResultsE, res, env);
    //     stringstream s;
    //     for(int i=0;i<res[0].size();i++)
    //         s<<res[0][i]<<" ";
    //     s<<endl;
    //     ofstream outfile("./debug-srhe2.txt",ios::out);
    //     outfile << s.str();
    //     outfile.close();
    // }
    // // Modulus switch
    // {
    //     #pragma omp parallel for
    //     for (size_t i = 0; i < intermediateResultsE.size(); i++) {
    //         mod_switch_to_next_inplace(intermediateResultsE[i], env);
    //     }
    // }

    //Debug Only
    // vector<vector<double>> res6;
    // decrypt(intermediateResultsE,res6,env);
    // encrypt(res6,intermediateResultsE,env,intermediateResultsE[0].scale);


    // FC Layer
    {
        start = std::chrono::steady_clock::now();
        vector<SealBfvCiphertext> resultE;
        vector<SealBfvPlaintext> WeightsP;
        vector<SealBfvPlaintext> BiasesP;
        vector<vector<double>>weight(100,vector<double>(env.poly_modulus_degree,0));
        vector<double> vMsgs(env.poly_modulus_degree,0.0);
        // encode(params.FCWeights,WeightsP,env,FC_SCALE);
        #pragma omp parallel for
        for(int n=0;n<10;n++){
            for(int i=0;i<10;i++){
                for(int j=0;j<256;j++){
                    if(i*256+j<50*7*7){
                        for(int k=0;k<BATCH_SIZE;k++){
                            weight[n*10+i][k*256+j]=params.FCWeights[n*50*7*7+i*256+j];
                        }
                    }else{break;}
                }
            }
        }
        for(int i=0;i<100;i++){
            WeightsP.emplace_back(move(SealBfvPlaintext(weight[i],env,FC_SCALE)));
        }

        for(int i=0;i<10;i++){
            for(int j=0;j<BATCH_SIZE;j++)
                vMsgs[j*256]=params.FCBiases[i];
            BiasesP.emplace_back(move(SealBfvPlaintext(vMsgs,env,intermediateResultsE[0].scale*FC_SCALE)));
        }

        fc(getPointers(intermediateResultsE),WeightsP,BiasesP,10,resultE,env);
        intermediateResultsE.clear();
        vector<SealBfvCiphertext>().swap(intermediateResultsE);
        intermediateResultsE=move(resultE);
        end = std::chrono::steady_clock::now();
        elapsed_seconds = end - start;

        cout<<"FC compute time: "<<elapsed_seconds.count()<<endl;
        
         // Modulus switch
        {
            #pragma omp parallel for
            for (size_t i = 0; i < intermediateResultsE.size(); i++) {
                mod_switch_to_next_inplace(intermediateResultsE[i], env);
            }
        }

        cout << "- Noise budget: " << env.environments[0].decryptorPtr->invariant_noise_budget(intermediateResultsE[0].eVectors[0]) << endl;
        
    }


    {
        start = std::chrono::steady_clock::now();
        srand((unsigned)time(NULL)); 
        double random=(1+(rand()%1000000))/1000000;
        SealBfvPlaintext random_value=SealBfvPlaintext(random,env,intermediateResultsE[0].scale);
        for(int i=0;i<intermediateResultsE.size();i++){
            add_plain_inplace(intermediateResultsE[i],random_value,env);
        }
        end = std::chrono::steady_clock::now();
        elapsed_seconds = end - start;
        cout<<"Random Layer compute time: "<<elapsed_seconds.count()<<endl;
        cout << "- Noise budget: " << env.environments[0].decryptorPtr->invariant_noise_budget(intermediateResultsE[0].eVectors[0]) << endl;
    }
    //debug
    // {
    //     vector<vector<double>> res;
    //     decrypt(intermediateResultsE, res, env);
    //     stringstream s;
    //     for(int i=0;i<res[0].size();i++)
    //         s<<res[0][i]<<" ";
    //     s<<endl;
    //     ofstream outfile("./debug-fc.txt",ios::out);
    //     outfile << s.str();
    //     outfile.close();
    // }
    
    // vector<vector<double>> res;
    // decrypt(intermediateResultsE, res, env);

    return intermediateResultsE;
}

int main()
{
    SealBfvEnvironment env = SealBfvEnvironment(
        POLY_MODULUS_DEGREE,
        PLAINTEXT_MODULUS);

    auto params = readParams("./resources/CifarReLUNoya.txt");
    vector<vector<vector<double>>> data;
    vector<vector<size_t>> labels;
    readInput(BATCH_SIZE, 1.0/255, data, labels,"./resources/CIFAR-3x32x32-test.txt");
    // Batch processing

    size_t correct = 0;
    for (size_t batchIndex = 0; batchIndex < data.size(); batchIndex++)
    {   
        cout<<"start a batch "<<batchIndex<<endl;
        //Conv Organization 
        vector<vector<vector<vector<double>>>>image_data(BATCH_SIZE,vector<vector<vector<double>>>(3, vector<vector<double>>(34, vector<double>(34,0.0))));
        // zero-pad image
        for(int d=0;d<3;d++){
            for (int i = 1; i <= 32; i++) {
                for (int j = 1; j <= 32; j++) {
                    for(int k=0;k<BATCH_SIZE;k++)
                        image_data[k][d][i][j] = data[batchIndex][k][d*32*32+(i - 1) * 32 + (j - 1)];
                }
            }
        }
        vector<vector<double>> msgs = ConvertToConvolutionRepresentationNew(image_data);
                    // for(int i=0;i<256*3;i++)
                    //     cout<<msgs[0][i]<<" ";
                    // cout<<endl;
        assert(msgs[0].size()==POLY_MODULUS_DEGREE);
        cout<<"finish ConvertToConvolutionRepresentation"<<endl;
        // Encrypt
        vector<SealBfvCiphertext> inputE;
        auto decor = make_decorator(encrypt, "Encryption");
        decor(msgs, inputE, env, INPUT_SCALE);
        cout << "- Noise budget: " << env.environments[0].decryptorPtr->invariant_noise_budget(inputE[0].eVectors[0]) << endl;

        // Forward pass on the cloud
        std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
        auto intermediateResultsE = Noya(params, inputE, env);
        std::chrono::time_point<std::chrono::steady_clock> end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        cout << setw(20) << "Total time: ";
        cout << elapsed_seconds.count() << " seconds" << std::endl;

        //output 10 ciphertext
        //Decrypt(user)
        vector<vector<double>> res;
        auto decorD=make_decorator(decrypt,"Decryption");
        decorD(intermediateResultsE, res, env);

        // for(int i=0;i<256*2;i++)
        //     cout<<res[0][i]<<" ";
        //     cout<<endl;
        // cout<<res.size()<<"  "<<res[0].size()<<endl;
        for(int i=0;i<BATCH_SIZE;i++){
            vector<double>val;
            for(int j=0;j<10;j++){
                double node=0;
                for(int k=0;k<256;k++)
                    node+=res[j][i*256+k];
                val.push_back(node);
                // cout<<node<<" ";
            }
            // cout<<endl;
            auto predicted = std::distance(val.begin(), std::max_element(val.begin(),val.end()));
            cout << labels[batchIndex][i] << " vs " << predicted<< endl;
            correct += labels[batchIndex][i] == predicted ? 1 : 0;
        }
        cout << "Accuracy is " << correct << "/" << ((batchIndex+1)*BATCH_SIZE);
        cout << "=" << (1.0 * correct)/((batchIndex+1)*BATCH_SIZE) << endl;
        // break;
    }
     cout << "Accuracy is " << correct << "/" << 10000;
        cout << "=" << (1.0 * correct)/10000 << endl;
    return 0;
}