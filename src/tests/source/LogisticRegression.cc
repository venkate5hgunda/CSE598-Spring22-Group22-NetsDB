#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include "PDBClient.h"

#include "FFMatrixBlock.h"
#include "FFMatrixUtil.h"
#include "SimpleFF.h"

using namespace std;

int main(int argc, char *argv[]) {
    // bool reloadData = true;
    string errMsg;
    string input_path, labels_path, w_path, b_path;
    int block_x, block_y, batch_size;
    int numFeatures, numNeurons, numLabels;
    if(argc<3) {
        cout << "Usage: blockDimensionX blockDimensionY batchSize numFeatures numNeurons numLabels"
                "path/to/weights/and/bias(leave empty if generate random)"
             << endl;
        exit(-1);
    }

    block_x = atoi(argv[1]);
    block_y = atoi(argv[2]);
    batch_size = atoi(argv[3]);
    numFeatures = atoi(argv[4]);
    numNeurons = atoi(argv[5]);
    numLabels = atoi(argv[6]);

    cout << "Using Block Dimensions " << block_x << ", " << block_y << endl;

    bool generate = true;
}