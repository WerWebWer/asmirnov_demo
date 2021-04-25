
#include <vector>
#include <memory>
#include <string>
#include <iterator>


#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <inference_engine.hpp>

using namespace InferenceEngine;

int main(int argc, char** argv) {
	//const std::string device_name{argv[1]};
	InferenceEngine::Core ie;

	CNNNetwork network = ie.ReadNetwork("../person-detection-retail-0002.xml",
					                    "../person-detection-retail-0002.bin");
	size_t batchSize = network.getBatchSize();

// -----------------
    InputInfo::Ptr input_info = network.getInputsInfo().begin()->second;
    std::string input_name = network.getInputsInfo().begin()->first;

    input_info->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
    input_info->setLayout(Layout::NHWC);
    input_info->setPrecision(Precision::U8);


    DataPtr output_info = network.getOutputsInfo().begin()->second;
    std::string output_name = network.getOutputsInfo().begin()->first;

	const SizeVector outputDims = output_info->getTensorDesc().getDims();
	const int numPred = outputDims[2];
	const int objectSize = outputDims[3];

	std::vector<size_t> imageWidths, imageHeights;

	output_info->setPrecision(Precision::FP32);
// -----------------

	ExecutableNetwork executable_network = ie.LoadNetwork(network, "CPU", {});
	InferRequest infer_request = executable_network.CreateInferRequest();

	cv::Mat image = cv::imread("../test.bmp");
	size_t w = image.cols;
	size_t h = image.rows;
// without wrapMat2Blob
	size_t channels = image.channels();
	size_t height = image.size().height;
	size_t width = image.size().width;

	InferenceEngine::TensorDesc tDesc( InferenceEngine::Precision::U8,
             				           { 1, 3, h, w },
             				           InferenceEngine::Layout::NCHW );
  
	Blob::Ptr imgBlob =  InferenceEngine::make_shared_blob<uint8_t>( tDesc, image.data );
// without wrapMat2Blob

	infer_request.SetBlob(input_name, imgBlob); 

	infer_request.Infer();

	Blob::Ptr output = infer_request.GetBlob(output_name); 
	// Print classification results
	// ClassificationResult classificationResult(output, {"../test.bmp"});
	// classificationResult.print();

// ----------------
	const Blob::Ptr output_blob = infer_request.GetBlob(output_name);
	MemoryBlob::CPtr moutput = as<MemoryBlob>(output_blob);
	if (!moutput) {
		throw std::logic_error("We expect output to be inherited from MemoryBlob, "
							   "but by fact we were not able to cast output to MemoryBlob");
	}
	auto moutputHolder = moutput->rmap();
	const float *detection= moutputHolder.as<const PrecisionTrait<Precision::FP32>::value_type *>();

	std::vector<float> probs;
	std::vector<cv::Rect> boxes;

	float score = 0;
	float cls = 0;
	float id = 0;

	for (int i=0; i < numPred; i++) {       
	auto image_id = static_cast<int>(detection[i * objectSize + 0]);
	if (image_id < 0) 
		break;

	float confidence = detection[i * objectSize + 2];
	auto label = static_cast<int>(detection[i * objectSize + 1]);
	auto xmin = static_cast<int>(detection[i * objectSize + 3] * imageWidths[image_id]);
	auto ymin = static_cast<int>(detection[i * objectSize + 4] * imageHeights[image_id]);
	auto xmax = static_cast<int>(detection[i * objectSize + 5] * imageWidths[image_id]);
	auto ymax = static_cast<int>(detection[i * objectSize + 6] * imageHeights[image_id]);
	if (confidence > 0.5) {
		boxes.push_back(cv::Rect(xmin , ymin,
								 xmax-xmin, 
								 ymax-ymin));
		}
	}
	cv::Mat result(image);
	for (int i = 0; i < numPred; i++) {
		cv::rectangle(result, boxes[i], cv::Scalar(0, 255, 0));
	}
	

	bool check = imwrite("/mnt/streltsova_openvino/Result.jpg", result);
	if (check == false) {
		std::cout << "Mission - Saving the image, FAILED" << std::endl;
	}



	return 0;
}