//
// Created by paulg on 1/16/2022.
//

#include "mex.h"
#include "vector"
#include "process_txtCPP.cpp"


void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] ) {


    mxAssert(nrhs == 1, "Only one input is expected, filename");
    mxAssert(nlhs == 1, "Only one output is expected, cell array");


    char * filename =  mxArrayToString(prhs[0]);
    vector<vector<dataPoint>> out;
    int nfields = 4;
    const char* fnames[] = {"name", "x", "y", "z"};

    processFile(filename, out);

    mwSize numDims = 1;
    int tmp = 0;
    if (!out.empty()){
        tmp = out[0].size();
    }
    mwSize dims[1] = {(mwSize) tmp};
    mxArray * outCells = mxCreateCellArray(numDims, dims);

    mwSize numDims2 = 1;
    mwSize dims2[1] = {(mwSize) out.size()};

    for (int i =0; i < dims[0]; i++){
        mxArray* structArrMX = mxCreateStructArray(numDims2, dims2, nfields, fnames);
        for (int j = 0; j < dims2[0]; j++){
            dataPoint datapoint = out[j][i];
            mxSetField(structArrMX, j, fnames[0], mxCreateString(datapoint.name.c_str()));
            mxSetField(structArrMX, j, fnames[1], mxCreateDoubleScalar(datapoint.X));
            mxSetField(structArrMX, j, fnames[2], mxCreateDoubleScalar(datapoint.Y));
            mxSetField(structArrMX, j, fnames[3], mxCreateDoubleScalar(datapoint.Z));
        }
        mxSetCell(outCells, i, structArrMX);
    }

    plhs[0] = outCells;

}
