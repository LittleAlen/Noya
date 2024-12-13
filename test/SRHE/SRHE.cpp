//the network we use is conv 25 conv 50 fc10

#include<stdlib.h>
#include<time.h>
#include<fstream>
#include <sstream>
#include<iostream>
#include "SealBfvVector.h"
#include "SealBfvCrtWrapper.h"
using namespace std;
using namespace seal;
using namespace mycryptonets;



namespace {
    void create_random_value(int n,vector<int>& sign,vector<double> &result){

        srand((unsigned)time(NULL)); 
        for(int i=0;i<n;i++)
        {
            if((rand()%2)==0)
                sign.emplace_back(1);
            else
                sign.emplace_back(-1);
        }
        // cout<<"in create_random_value sign 0: "<<sign[0]<<endl;
        for(int i=0; i<n; i++) 
        { 
            result.emplace_back(sign[i]*abs(1+rand()%100));
            // result.emplace_back(sign[i]*(1));
        }
        // cout<<"in create_random_value result 0: "<<result[0]<<endl;
    }



    void server_multi_plain_value( const vector<SealBfvCiphertext> &data, const vector<SealBfvPlaintext>& plain_random_value,vector<SealBfvCiphertext> &result,const SealBfvEnvironment & env)
    {
        SealBfvCiphertext temp;
        for(int i=0;i<data.size();i++)
        {
            multiply_plain(data[i], plain_random_value[i], temp, env);
            result.emplace_back(move(temp));
        }
    }



    void client_decrypt_random_value(vector<SealBfvCiphertext> &data,vector<vector<int>> &result_sign,const SealBfvEnvironment & env)
    {
        std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
        vector<vector<double>> res;
        decrypt(data, res, env);
        for(int i=0;i<res.size();i++)//client compute the random value's sign which positive sign is 1, negative is -1,zero is 0
        {
            result_sign.emplace_back(vector<int>());
            for(int j=0;j<res[0].size();j++)
            {
                double v=res[i][j];
                if(v>0.0000000000001)
                    result_sign[i].emplace_back(1);
                else if(v<-0.0000000000001)
                    result_sign[i].emplace_back(-1);
                else
                    result_sign[i].emplace_back(0);
            }
        }
        std::chrono::time_point<std::chrono::steady_clock> end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
    }

    void simulate_relu(vector<SealBfvCiphertext> & intermediateResultsE,const SealBfvEnvironment & env){
        // replace to SRHE layer(Simulative Relu with Homomorphic Encrytion)
    {
        vector<SealBfvCiphertext> resultE;
        vector<SealBfvPlaintext>  plain_relu;
        
        vector<int> sign;
        vector<vector<int>> result_sign;
        
        std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
        
        {
        
        vector<double> random_value;
        vector<SealBfvPlaintext>  plain_random_value;
        vector<SealBfvCiphertext> middle_value;
        create_random_value(intermediateResultsE.size(),sign,random_value);
        for(int i=0;i<random_value.size();i++)
        {
            //1 plain_random_value.emplace_back(move(SealBfvPlaintext(random_value[i],env)));
            SealBfvCiphertext tmp(1);
            multiply_plain(intermediateResultsE[i],SealBfvPlaintext(random_value[i],env),tmp,env);
            middle_value.emplace_back(move(tmp));
        }
        // std::chrono::time_point<std::chrono::steady_clock> middle = std::chrono::steady_clock::now();
        // std::chrono::duration<double> seconds = middle - start;
        //cout<<"Create random value times:  "<<seconds.count()<<"s"<<endl;//too small 

        // 1 server_multi_plain_value(intermediateResultsE,plain_random_value,middle_value,env);
        

        client_decrypt_random_value(middle_value,result_sign,env);
        }
       

        for(int i=0;i<sign.size();i++){
            for(int j=0;j<result_sign[0].size();j++){
                result_sign[i][j]*=sign[i];
                if (result_sign[i][j]<=0)
                    result_sign[i][j]=0;
                else
                    result_sign[i][j]=1;
            }
        }



        vector<int> valid;//we need to make a check, because when all of the number is 0, plaintext is make no sense, the SEAL library will throw a transparent exception.
        for(int i=0;i<sign.size();i++){
            int test=0;
            for(int j=0;j<result_sign[0].size();j++){
                if (result_sign[i][j]>0){
                    test=1;
                    break;
                }
            }
            valid.push_back(test);
        }
        for(int i=0;i<sign.size();i++){
            plain_relu.emplace_back(move(SealBfvPlaintext(result_sign[i],env)));
        }
        

        for(int i=0;i<intermediateResultsE.size();i++){
            if(valid[i]==0)
            {
                resultE.emplace_back(move(SealBfvCiphertext(intermediateResultsE[i].scale)));
                continue;
            }
            // SealBfvCiphertext temp;
            // multiply_plain(intermediateResultsE[i],plain_relu[i],temp,env);
            // resultE.emplace_back(move(temp));
            multiply_plain(intermediateResultsE[i],plain_relu[i],intermediateResultsE[i],env);
        }
        std::chrono::time_point<std::chrono::steady_clock> end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;

        // intermediateResultsE.clear();
        // intermediateResultsE = move(resultE); 
        
        int valid_index=0;
        while(intermediateResultsE[valid_index].batchSize==0)
            valid_index++;
    }
    }
}
namespace {
    size_t POLY_MODULUS_DEGREE = 8192; 
    size_t BATCH_SIZE=32;
  
    vector<Modulus> PLAINTEXT_MODULUS = PlainModulus::Batching(
    POLY_MODULUS_DEGREE, {18, 18});
    
    double INPUT_SCALE = 8*2;
    double CONV1_SCALE = 32*8;
    double CONV2_SCALE = 32*16;
    double FC_SCALE = 32*32;
    
   
}


double Test(SealBfvEnvironment &env , double v)
{   
    cout<<"------------------------------"<<endl;

    cout<<"PolyModulus Degree "<<env.poly_modulus_degree<<endl;
    vector<double> val(env.poly_modulus_degree,v);
    std::chrono::time_point<std::chrono::steady_clock> start, end;
    SealBfvCiphertext tmp(val,env,1);
    vector<SealBfvCiphertext> vec_tmp;
    vec_tmp.push_back(tmp);
    vector<SealBfvCiphertext> vec_tmp1=vec_tmp;
    std::chrono::duration<double> elapsed_seconds;

    
    auto noise1= env.environments[0].decryptorPtr->invariant_noise_budget(vec_tmp[0].eVectors[0]);
    cout<<"SRHE  Time: ";
    start=std::chrono::steady_clock::now();
    simulate_relu(vec_tmp,env);
    end=std::chrono::steady_clock::now();
    elapsed_seconds = end - start;
    cout<<elapsed_seconds.count()<<"  ";
    cout<<"Noise: "<<noise1-env.environments[0].decryptorPtr->invariant_noise_budget(vec_tmp[0].eVectors[0])<<endl;
    
    auto time1=elapsed_seconds.count();
    
    cout<<"Square  Time: ";
    noise1= env.environments[0].decryptorPtr->invariant_noise_budget(vec_tmp1[0].eVectors[0]);
    start=std::chrono::steady_clock::now();
    square_inplace_vec(vec_tmp1,env);
    end=std::chrono::steady_clock::now();
    elapsed_seconds = end - start;
    cout<<elapsed_seconds.count()<<"  ";
    cout<<"Noise: "<<noise1-env.environments[0].decryptorPtr->invariant_noise_budget(vec_tmp1[0].eVectors[0])<<endl;
    auto time2=elapsed_seconds.count();
    cout<<"rate "<<time1/time2<<endl;
    cout<<"------------------------------"<<endl;
    return (double)(time1/time2);
}

int main()
{
    //vector<double>tmp();
    SealBfvEnvironment env = SealBfvEnvironment(
        8192,
        PLAINTEXT_MODULUS);  
    int num=1000;
    double sum=0;  
    for(int i=0;i<100000000;i+=100000){
        sum+=Test(env,i);
    }
    cout<<"Total rate :"<<sum/num<<endl;
   
    return 0;
}