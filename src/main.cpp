#include "opencv2/surface_matching.hpp"
#include <iostream>
#include "opencv2/surface_matching/ppf_helpers.hpp"
#include "opencv2/core/utility.hpp"




using namespace std;
using namespace cv;
using namespace ppf_match_3d;



int main()
{
    // welcome message
    cout << "****************************************************" << endl;
    cout << "* Surface Matching demonstration : demonstrates the use of surface matching"
             " using point pair features." << endl;
    cout << "* The sample loads a model and a scene, where the model lies in a different"
             " pose than the training.\n* It then trains the model and searches for it in the"
             " input scene. The detected poses are further refined by ICP\n* and printed to the "
             " standard output." << endl;
    cout << "****************************************************" << endl;

    string modelFileName = "model3Res2.ply";
    string sceneFileName = "sceneRes.ply";


    Mat pc = loadPLYSimple(modelFileName.c_str(), 1);

    // Now train the model
    cout << "Training..." << endl;
    int64 tick1 = cv::getTickCount();
    ppf_match_3d::PPF3DDetector detector(0.015, 0.05, 15);
    detector.trainModel(pc);
    int64 tick2 = cv::getTickCount();
    cout << endl << "Training complete in "
         << (double)(tick2-tick1)/ cv::getTickFrequency()
         << " sec" << endl << "Loading model..." << endl;

    // Read the scene
    Mat pcTest = loadPLYSimple(sceneFileName.c_str(), 1);

    // Match the model to the scene and get the pose
    cout << endl << "Starting matching..." << endl;
    vector<Pose3DPtr> results;
    tick1 = cv::getTickCount();
    detector.match(pcTest, results, 1.0/5.0, 0.05);
    tick2 = cv::getTickCount();
    cout << endl << "PPF Elapsed Time " <<
         (tick2-tick1)/cv::getTickFrequency() << " sec" << endl;

    // Get only first N results
    int N = 10;
    vector<Pose3DPtr> resultsSub(results.begin(),results.begin()+N);

    // Create an instance of ICP
    ICP icp(100, 0.005f, 2.5f, 8);
    int64 t1 = cv::getTickCount();

    // Register for all selected poses
    cout << endl << "Performing ICP on " << N << " poses..." << endl;
    icp.registerModelToScene(pc, pcTest, resultsSub);
    int64 t2 = cv::getTickCount();

    cout << endl << "ICP Elapsed Time " <<
         (t2-t1)/cv::getTickFrequency() << " sec" << endl;

    cout << "Poses: " << endl;
    string outputname;
    float minRes = 1;
    int minResIdx;
    // debug first five poses
    for (size_t i=0; i<resultsSub.size(); i++)
    {
        Pose3DPtr result = resultsSub[i];
        double residual = result[0].residual;
        if(residual<minRes){
        	minRes = residual;
        	minResIdx = i;
        }

        cout << "Pose Result " << i << endl;
        result->printPose();
        //cout<<"Computed Residual : "<<residual<<endl;
		if (residual < 0.01) {
			outputname = "result0" + to_string(i) + ".ply";
			int strlen = outputname.length();
			char char_array[strlen + 1];
			strcpy(char_array, outputname.c_str());
			Mat pct = transformPCPose(pc, result->pose);
			writePLY(pct, char_array);
        }
    }

	Mat pctBest = transformPCPose(pc, resultsSub[minResIdx]->pose);
	writePLY(pctBest, "resultBest.ply");


    return 0;

}
