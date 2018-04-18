package com.nanamare.mac.mobilenet;

import android.graphics.Bitmap;

public class MobileNetNCNN {

    public native boolean init(byte[] params, byte[] bin, byte[] labels);
    public native String inference(Bitmap bitmap);
    public native String stringFromJNI();

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("objectclassification");
    }

}
