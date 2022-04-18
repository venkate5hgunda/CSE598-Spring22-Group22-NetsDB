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
    bool reloadData = true; // To toggle if we need randomly generated data to test
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
    numNeurons = atoi(argv[5]); // For LogReg, numFeatures and numNeurons can be made same
    numLabels = atoi(argv[6]);

    cout << "Using Block Dimensions " << block_x << ", " << block_y << endl;

    bool generate = false; // I want to reload existing weights, biases from TF

    string masterIp = "localhost";
    pdb::PDBLoggerPtr clientLogger = make_shared<pdb::PDBLogger>("FFClientLog");
    pdb::PDBClient pdbClient(8108, masterIp, clientLogger, false, true);
    pdb::CatalogClient catalogClient(8108, masterIp, clientLogger);

    if(reloadData) {
        ff::createDatabase(pdbClient, "ff");
        ff::setup(pdbClient, "ff");

        ff::createSet(pdbClient, "ff", "inputs", "inputs", 64);
        ff::createSet(pdbClient, "ff", "label", "label", 64);

        ff::createSet(pdbClient, "ff", "w", "W", 64);
        ff::createSet(pdbClient, "ff", "b", "B", 64);
    }

    ff::createSet(pdbClient, "ff", "output", "Output", 256);
    ff::createSet(pdbClient, "ff", "y", "Y", 64);

    if(!generate && reloadData) { // First time, we reload data from .out files
        input_path = string(argv[4]) + "/input.out";
        labels_path = string(argv[4]) + "/labels.out";
        w_path = string(argv[4]) + "/w.out";
        b_path = string(argv[4]) + "/b.out";

        ff::load_matrix_data(pdbClient, input_path, "ff", "inputs", block_x, block_y, false, false, errMsg);
        (void)ff::load_matrix_data(pdbClient, w_path, "ff", "w", block_x, block_y, false, false, errMsg);
        (void)ff::load_matrix_data(pdbClient, b_path, "ff", "b", block_x, block_y, false, false, errMsg);
    } else if (reloadData) {
        std::cout << "To load matrix for ff:inputs" << std::endl;
        ff::loadMatrix(pdbClient, "ff", "inputs", batch_size, numFeatures, block_x, block_y, false, false, errMsg);

        std::cout << "To load matrix for ff:w" << std::endl;
        ff::loadMatrix(pdbClient, "ff", "w", numLabels, numNeurons, block_x, block_x, false, false, errMsg);
        // 2 x 1
        std::cout << "To load matrix for ff:b" << std::endl;
        ff::loadMatrix(pdbClient, "ff", "b", numLabels, 1, block_x, 1, false, true, errMsg);
    }

    double dropout_rate = 0.5;

    auto begin = std::chrono::high_resolution_clock::now();
    ff::inference_unit(pdbClient, "ff", "w", "inputs", "b", "output", dropout_rate); // Need to write my own for LogReg now
    auto begin = std::chrono::high_resolution_clock::now();
    std::cout << "*****FFTest End-to-End Time Duration: ****" << std::chrono::duration_cast<std::chrono::duration<float>>(end - begin).count() << " secs." << std::endl;

    vector<vector<double>> labels_test;

    if(!generate) {
        ff::load_matrix_from_file(labels_path, labels_test);
    }

    int total_count = 0;
    int correct_count = 0;
    {
        pdb::UseTemporaryAllocationBlock tempBlock{1024*1024*128};

        auto iterator = pdbClient.getSetIterator<FFMatrixBlock>("ff", "output");

        for(auto r:iterator) {
            total_count++;
            double *data = r->getRawDataHandle()->c_ptr();
            int i = 0;
            int j = r->getBlockRowIndex() * r->getRowNums();
            while (i < r->getRowNums() * r->getColNums()) {
                if (!generate && j >= labels_test.size())
                    break;

                cout << data[i] << ", " << data[i + 1] << endl;

                if (!generate) {
                    int pos1 = data[i] > data[i + 1] ? 0 : 1;
                    int pos2 = labels_test[j][0] > labels_test[j][1] ? 0 : 1;

                    if (pos1 == pos2)
                        correct_count++;
                }

                i += r->getColNums();
                j++;
            }
        }

        if (!generate)
            cout << "Accuracy: " << correct_count << "/" << labels_test.size() << std::endl;
    }

    std::cout << "count=" << total_count << std::endl;

    sleep(20);

    return 0;
}