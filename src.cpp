#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>
#include <string>
//person-detection-retail-0002
using namespace InferenceEngine;

static InferenceEngine::Blob::Ptr wrapMat2Blob( const cv::Mat &mat ) {
	size_t channels = mat.channels();
	size_t height = mat.size().height;
	size_t width = mat.size().width;
  
	size_t strideH = mat.step.buf[0];
	size_t strideW = mat.step.buf[1];
  
	bool is_dense = (strideW == channels && strideH == channels * width);
	if( !is_dense ) THROW_IE_EXCEPTION << "Doesn't support conversion from not dense cv::Mat";
 
	InferenceEngine::TensorDesc tDesc( InferenceEngine::Precision::U8, { 1, channels, height, width }, InferenceEngine::Layout::NCHW );
	return InferenceEngine::make_shared_blob<uint8_t>( tDesc, mat.data );
 }

int main(int argc, char** argv) {
	std::cout << "v1.0.1" << std::endl;
    	//const std::string device_name{argv[1]};
	InferenceEngine::Core ie;
//person-detection-retail-0002
	CNNNetwork network = ie.ReadNetwork("../model/person-detection-retail-0002.xml");
	size_t batchSize = network.getBatchSize();

// -----------------
    InputInfo::Ptr input_info = network.getInputsInfo().begin()->second;
    std::string input_name = network.getInputsInfo().begin()->first;

    input_info->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
    input_info->setLayout(Layout::NCHW);
    input_info->setPrecision(Precision::U8);


    DataPtr output_info = network.getOutputsInfo().begin()->second;
    std::string output_name = network.getOutputsInfo().begin()->first;

	const SizeVector outputDims = output_info->getTensorDesc().getDims();
	const int numPred = outputDims[2];
	const int objectSize = outputDims[3];

	std::vector<size_t> imageWidths, imageHeights;

	output_info->setPrecision(Precision::FP32);
// -----------------

	ExecutableNetwork executable_network = ie.LoadNetwork(network, "ARM");
	InferRequest infer_request = executable_network.CreateInferRequest();

	cv::Mat image = cv::imread("../photo/test1.jpg");
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
std::cout << "here2" << std::endl;

	

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


    std::vector<cv::Rect> boxes;
    cv::Mat result(image);
std::cout << "here" << std::endl;

    for (int i = 0; i < numPred; i++) {       
		auto image_id = static_cast<int>(detection[i * 7 + 0]);
		if (image_id < 0) 
			break;

		float confidence = detection[i * 7 + 2];
		auto label = static_cast<int>(detection[i * 7 + 1]);
		auto xmin = static_cast<int>(detection[i * 7 + 3] * w);
		auto ymin = static_cast<int>(detection[i * 7 + 4] * h);
		auto xmax = static_cast<int>(detection[i * 7 + 5] * w);
		auto ymax = static_cast<int>(detection[i * 7 + 6] * h);
		if (confidence > 0.5) {
			cv::rectangle(result, cv::Rect(xmin , ymin, xmax-xmin, ymax-ymin), cv::Scalar(0, 255, 0));
		}
	}


    bool check = imwrite("test_out.jpg", result);
    if (check == false) {
        std::cout << "Mission - Saving the image, FAILED" << std::endl;
    }



    return 0;
}