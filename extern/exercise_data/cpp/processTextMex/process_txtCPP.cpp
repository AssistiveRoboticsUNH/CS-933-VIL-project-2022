//
// Created by paulg on 1/16/2022.
//

#include <string>
#include "vector"
#include <iostream>
#include <fstream>
#include "sstream"

using namespace std;

struct dataPoint{
    string name;
    double X;
    double Y;
    double Z;
};

void skipLines(ifstream* myfile, int skipLines){
    string line;
    while (skipLines > 0 && getline(*myfile,line)) {
        skipLines--;
    }
}

void processFile(const char* filename,  vector<vector<dataPoint>>& out){

    ifstream myfile;
    myfile.open(filename);
    skipLines(&myfile, 9);

    string line;
    getline(myfile,line);

    stringstream ss(line);
    vector<string> names;
    getline(ss, line, '\t');
    while (getline(ss, line, '\t')){
        names.push_back(line);
    }

    while (getline(myfile,line)) {
        ss = stringstream(line);
        vector<dataPoint> dataPointVec;
        for (int i = 0; i < names.size(); i++){
            dataPoint dp;
            string name = names[i];
            dp.name = name;
            ss >> dp.X;
            ss >> dp.Y;
            ss >> dp.Z;
            dataPointVec.push_back(dp);

        }
        out.push_back(dataPointVec);

    }
    myfile.close();
}

int main(int num, const char** argsv){

//    string filename =  "D:/NSFExercise/data/Subject 18/Subject 18 Order 1L.txt";
    string filename =  "D:/NSFExercise/data/Subject 13/Subject 13 Order2L0001.tsv";
    vector<vector<dataPoint>> out;
    processFile(filename.c_str(), out);
    return 0;
}
