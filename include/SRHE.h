#pragma once
#include<stdlib.h>
#include<time.h>
#include<fstream>
#include <sstream>
#include<iostream>
#include "SealBfvVector.h"
#include "SealBfvCrtWrapper.h"

namespace mycryptonets{
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
        cout<<"client compute time: "<<elapsed_seconds.count()<<endl;
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
        cout<<"SRHE times:  "<<elapsed_seconds.count()<<"s"<<endl;

        // intermediateResultsE.clear();
        // intermediateResultsE = move(resultE); 
        
        int valid_index=0;
        while(intermediateResultsE[valid_index].batchSize==0)
            valid_index++;
        cout << "- Noise budget: " << env.environments[0].decryptorPtr->invariant_noise_budget(intermediateResultsE[valid_index].eVectors[0]) << endl;

    }
    }
}