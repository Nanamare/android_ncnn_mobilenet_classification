#include <jni.h>

#include <android/bitmap.h>
#include <android/log.h>

#include <net.h>
#include <sys/time.h>
#include <unistd.h>

#include <string>
#include <vector>

#include "mobilenet.id.h"

static int MOBILENET_INPUT_SIZE = 224;

static std::vector<unsigned char> mobileNetParams;
static std::vector<unsigned char> mobileNetBin;
static std::vector<std::string> mobileNetLabels;
static ncnn::Net mobileNet;

static struct timeval startTime;
static struct timeval endTime;
static double elasped;

extern "C"
JNIEXPORT jstring JNICALL
Java_com_nanamare_mac_mobilenet_MobileNetNCNN_stringFromJNI(JNIEnv *env, jobject instance) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}

static void bench_start() {
    gettimeofday(&startTime, NULL);
}

static void bench_end(const char *comment) {
    gettimeofday(&endTime, NULL);
    elasped = ((endTime.tv_sec - startTime.tv_sec) * 1000000.0f + endTime.tv_usec -
               startTime.tv_usec) / 1000.0f;
    __android_log_print(ANDROID_LOG_DEBUG, "MobileNet_V1", "%.2fms   %s", elasped, comment);
}


static std::vector<std::string> split_string(const std::string &str, const std::string &delimiter) {
    std::vector<std::string> strings;

    std::string::size_type pos = 0;
    std::string::size_type prev = 0;
    while ((pos = str.find(delimiter, prev)) != std::string::npos) {
        strings.push_back(str.substr(prev, pos - prev));
        prev = pos + 1;
    }

    // To get the last substring (or only, if delimiter is not found)
    strings.push_back(str.substr(prev));

    return strings;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_nanamare_mac_mobilenet_MobileNetNCNN_init(JNIEnv *env, jobject instance,
                                                   jbyteArray params_, jbyteArray bin_,
                                                   jbyteArray labels_) {
    jbyte *params = env->GetByteArrayElements(params_, NULL);
    jbyte *bin = env->GetByteArrayElements(bin_, NULL);
    jbyte *labels = env->GetByteArrayElements(labels_, NULL);

    {
        int len = env->GetArrayLength(params_);
        mobileNetParams.resize(len);
        env->GetByteArrayRegion(params_, 0, len, (jbyte *) mobileNetParams.data());
        int ret = mobileNet.load_param(mobileNetParams.data());
        __android_log_print(ANDROID_LOG_DEBUG, "MobileNet params", "load params %d %d", ret, len);
    }

    {
        int len = env->GetArrayLength(bin_);
        mobileNetBin.resize(len);
        env->GetByteArrayRegion(bin_, 0, len, (jbyte *) mobileNetBin.data());
        int ret = mobileNet.load_model(mobileNetBin.data());
        __android_log_print(ANDROID_LOG_DEBUG, "MobileNet binary", "load binary %d %d", ret, len);
    }

    {
        int len = env->GetArrayLength(labels_);
        std::string words_buffer;
        words_buffer.resize(len);
        env->GetByteArrayRegion(labels_, 0, len, (jbyte *) words_buffer.data());
        mobileNetLabels = split_string(words_buffer, "\n");
    }

    env->ReleaseByteArrayElements(params_, params, 0);
    env->ReleaseByteArrayElements(bin_, bin, 0);
    env->ReleaseByteArrayElements(labels_, labels, 0);

    return JNI_TRUE;


}


extern "C"
JNIEXPORT jstring JNICALL
Java_com_nanamare_mac_mobilenet_MobileNetNCNN_inference(JNIEnv *env, jobject instance,
                                                        jobject bitmap) {

    bench_start();

    ncnn::Mat input;
    {
        AndroidBitmapInfo androidBitmapInfo;
        AndroidBitmap_getInfo(env, bitmap, &androidBitmapInfo);

        int width = androidBitmapInfo.width;
        int height = androidBitmapInfo.height;

        if (width != MOBILENET_INPUT_SIZE || height != MOBILENET_INPUT_SIZE) {
            return NULL;
        }

        if (androidBitmapInfo.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
            return NULL;
        }

        void *pixelData;

        AndroidBitmap_lockPixels(env, bitmap, &pixelData);

        input = ncnn::Mat::from_pixels((const unsigned char *) pixelData, ncnn::Mat::PIXEL_RGBA2BGR,
                                       width, height);

        AndroidBitmap_unlockPixels(env, bitmap);
    }

    std::vector<float> scoreList;
    {
        //mean subtraction
        const float meanValues[3] = {103.94f, 116.78f, 123.68f}; // mobileNet mean value RGB 103.94f, 116.78f, 123.68f
        input.substract_mean_normalize(meanValues, 0);

        ncnn::Extractor extractor = mobileNet.create_extractor();
        extractor.set_light_mode(true);
        extractor.set_num_threads(4);

        extractor.input(mobilenet_param_id::BLOB_data, input);

        ncnn::Mat output;
        extractor.extract(mobilenet_param_id::BLOB_prob, output);

        scoreList.resize(static_cast<unsigned long>(output.c));
        __android_log_print(ANDROID_LOG_DEBUG, "Output width, ScoreList size", "%d  %d", output.c, static_cast<int>(scoreList.size()));
        for (int index = 0; index < output.c; index++)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "Output probability", "%d %f", index ,static_cast<float>(output[index]));
            scoreList[index] = output[index];
        }

        /* gcc extension
        scoreList.resize(output.c);
        for (int j=0; j<output.c; j++)
        {
            const float* prob = output.data + output.cstep * j;
            scoreList[j] = prob[0];
        }
        */
    }

    int top_class = 0;
    float max_score = 0.f;
    for (size_t i = 0; i < scoreList.size(); i++)
    {
        float s = scoreList[i];
        if (s > max_score)
        {
            top_class = static_cast<int>(i);
            max_score = s;
            __android_log_print(ANDROID_LOG_DEBUG, "top_class max_score", "%d %f", top_class ,max_score);
        }
    }


    const std::string& word = mobileNetLabels[top_class];
    char tmp[32];

    sprintf(tmp, "%.3f", max_score);
    std::string result_str = std::string(word.c_str() + 10) + " = " + tmp;

    // +10 to skip leading n03179701(Remove prefix)
    jstring result = env->NewStringUTF(result_str.c_str());

    bench_end("Detect");

    return result;
}
