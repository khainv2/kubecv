package com.khainv9.kube;
import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Build;
import android.os.Bundle;

import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvException;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import org.opencv.photo.Photo;
import static org.opencv.core.CvType.CV_8UC3;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "MainActivity";
    JavaCameraView javaCameraView;
    ImageView ivResult;
    Mat src, dst, bilateralImage, denoisedImage, hsvImage, frameThreshed, imgray, hierarchy;
    boolean onCapture = false;
    public static final int CAMERA_PERMISSION_REQUEST_CODE = 3;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            if (status == LoaderCallbackInterface.SUCCESS)
                javaCameraView.enableView();
            else
                super.onManagerConnected(status);
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        if (ContextCompat.checkSelfPermission(this,Manifest.permission.CAMERA)!= PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,new String[]{Manifest.permission.CAMERA},CAMERA_PERMISSION_REQUEST_CODE);
        }
        LinearLayout myLayout = findViewById(R.id.mainLayout);

        javaCameraView = new JavaCameraView(this, 0);
        javaCameraView.setVisibility(View.VISIBLE);
        javaCameraView.setCvCameraViewListener(this);
        javaCameraView.setLayoutParams(new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.MATCH_PARENT));
        myLayout.addView(javaCameraView);

        ivResult = new ImageView(this);
        ivResult.setVisibility(View.GONE);
        ivResult.setLayoutParams(new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.MATCH_PARENT));
        myLayout.addView(ivResult);

        Button btCapture = findViewById(R.id.capture);
        btCapture.setOnClickListener(view -> capture());
        Button btReset = findViewById(R.id.reset);
        btReset.setOnClickListener(view -> reset());
    }

    void capture(){
        onCapture = true;

        Log.d("capture", "Mat src " + src.type());

        Imgproc.cvtColor(src, dst,Imgproc.COLOR_RGBA2BGR);
        Imgproc.bilateralFilter(dst, bilateralImage, 9, 75, 75);
        Photo.fastNlMeansDenoisingColored(bilateralImage, denoisedImage, 10, 10, 7, 21);
        Imgproc.cvtColor(denoisedImage, hsvImage, Imgproc.COLOR_BGR2HSV);

        Scalar colorMin = new Scalar(20, 100, 100);
        Scalar colorMax = new Scalar(30, 255, 255);
        Core.inRange(hsvImage, colorMin, colorMax, frameThreshed);
        Mat imgray = frameThreshed.clone();
        java.util.List<MatOfPoint> contours = new java.util.ArrayList<>();
        Imgproc.findContours(imgray, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

        for (int i = 0; i < contours.size(); i++) {
            MatOfPoint contour = contours.get(i);
            org.opencv.core.Rect rect = Imgproc.boundingRect(contour);
            int x = rect.x;
            int y = rect.y;
            int w = rect.width;
            int h = rect.height;

            Log.d("get rect", "rect" + x + ", " + y + ", " + w + ", " + h);
            Imgproc.rectangle(src, new org.opencv.core.Point(x, y), new org.opencv.core.Point(x + w, y + h), new Scalar(0, 255, 0), 2);
        }

        Point point1 = new Point(100, 100);
        Point point2 = new Point(500, 300);
        Scalar color = new Scalar(64, 64, 64);
        int thickness = 10;
        Imgproc.rectangle (src, point1, point2, color, thickness);

        try {
            Bitmap bmp = Bitmap.createBitmap(src.cols(), src.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(src, bmp);
            ivResult.setImageBitmap(bmp);
        } catch (CvException e){
            Log.d("Exception", e.getMessage());
        }

        javaCameraView.setVisibility(View.GONE);
        ivResult.setVisibility(View.VISIBLE);
    }

    void reset(){
        onCapture = false;
        javaCameraView.setVisibility(View.VISIBLE);
        ivResult.setVisibility(View.GONE);
    }

    @Override
    protected void onPause(){
        super.onPause();
        if(javaCameraView!=null)
            javaCameraView.disableView();
    }

    @Override
    protected void onDestroy(){
        super.onDestroy();
        if(javaCameraView!=null)
            javaCameraView.disableView();
    }

    @Override
    protected void  onResume(){
        super.onResume();
        if (OpenCVLoader.initDebug()) {
            Log.i(TAG, "OpenCV loaded successfully.");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        } else {
            Log.i(TAG, "OpenCV not loaded.");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_1_0, this, mLoaderCallback);
        }
    }

    @Override
    public void onCameraViewStopped() {
        src.release();
    }

    @RequiresApi(api = Build.VERSION_CODES.M)
    void getPermission()
    {
        if(checkSelfPermission(Manifest.permission.CAMERA)!= PackageManager.PERMISSION_GRANTED)
        {
            requestPermissions(new String[]{Manifest.permission.CAMERA},101);
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.M)
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if(requestCode == 101 && grantResults.length > 0) {
            if(grantResults[0]!=PackageManager.PERMISSION_GRANTED){
                getPermission();
            }
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        src = new Mat(height, width, CV_8UC3);

        dst = new Mat();
        bilateralImage = new Mat();
        denoisedImage = new Mat();
        hsvImage = new Mat();
        frameThreshed = new Mat();
        imgray = new Mat();
        hierarchy = new Mat();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        if (onCapture){
            return inputFrame.rgba();
        } else {
            Mat mat = inputFrame.rgba();
            src = mat.clone();
            return src;
        }
    }
}
