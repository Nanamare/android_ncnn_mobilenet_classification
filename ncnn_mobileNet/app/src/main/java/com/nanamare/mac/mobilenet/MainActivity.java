package com.nanamare.mac.mobilenet;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.media.Image;
import android.media.ImageReader;
import android.net.Uri;
import android.os.SystemClock;
import android.os.Trace;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.LinkedList;
import java.util.List;

public class MainActivity extends CameraActivity implements ImageReader.OnImageAvailableListener {


    private static final Size DESIRED_PREVIEW_SIZE = new Size(1280, 720);
    private static final boolean MAINTAIN_ASPECT = true;


    private final String TAG = getClass().getSimpleName().trim();

    private static final int SELECT_IMAGE = 1;

    private TextView infoResult;
    private ImageView imageView;
    private Bitmap yourSelectedImage = null;

    private int[] rgbBytes = null;
    private int sensorOrientation;

    private Matrix frameToCropTransform;

    private Bitmap rgbFrameBitmap = null;
    private Bitmap resizeBitmap = null;


    @Override
    protected void onPreviewSizeChosen(Size size, int rotation) {

        int cropSize = 224;

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        sensorOrientation = rotation + 270;

        rgbBytes = new int[previewWidth * previewHeight];

        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888);

        resizeBitmap = Bitmap.createBitmap(cropSize, cropSize, Bitmap.Config.ARGB_8888);

        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        cropSize, cropSize,
                        sensorOrientation, MAINTAIN_ASPECT);

        yuvBytes = new byte[3][];

    }

    @Override
    public void onImageAvailable(final ImageReader reader) {

        Image image = null;

        try {
            image = reader.acquireLatestImage();

            if (image == null) {
                return;
            }

            Trace.beginSection("imageAvailable");

            final Image.Plane[] planes = image.getPlanes();
            fillBytes(planes, yuvBytes);


            // No mutex needed as this method is not reentrant.
            if (computing) {
                image.close();
                return;
            }
            computing = true;

            final int yRowStride = planes[0].getRowStride();
            final int uvRowStride = planes[1].getRowStride();
            final int uvPixelStride = planes[1].getPixelStride();

            ImageUtils.convertYUV420ToARGB8888(
                    yuvBytes[0],
                    yuvBytes[1],
                    yuvBytes[2],
                    previewWidth,
                    previewHeight,
                    yRowStride,
                    uvRowStride,
                    uvPixelStride,
                    rgbBytes);

            image.close();

        } catch (final Exception e) {
            if (image != null) {
                image.close();
            }
            Trace.endSection();
            return;
        }

        rgbFrameBitmap.setPixels(rgbBytes, 0, previewWidth, 0, 0, previewWidth, previewHeight);


        final Canvas canvas = new Canvas(resizeBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);

        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {

                        long startTime = SystemClock.uptimeMillis();
                        String result = mobileNetNcnn.inference(resizeBitmap);
                        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
                        Log.d("NCNN Inference time cost: ", String.valueOf(lastProcessingTimeMs));
                        Log.d("NCNN  Label : ", result);

                        computing = false;
                    }
                });

        Trace.endSection();
    }

    @Override
    protected int getLayoutId() {
        return R.layout.camera_fragment;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

}
