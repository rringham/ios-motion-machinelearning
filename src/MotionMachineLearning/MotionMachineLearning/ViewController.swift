//
//  ViewController.swift
//  MotionMachineLearning
//
//  Created by Rob Ringham on 1/4/16.
//  Copyright © 2016 Ringham. All rights reserved.
//

import UIKit
import CoreMotion

class ViewController: UIViewController {
    
    private let neuralNetwork : NeuralNetwork = NeuralNetwork(weightsFileName: "motion_nn")
    private let motionManager = CMMotionManager()
    private var timer: NSTimer?

    @IBOutlet weak var orientationLabel: UILabel!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        self.view.backgroundColor = UIColor.orangeColor()
        
        guard self.motionManager.accelerometerAvailable && self.motionManager.gyroAvailable && self.motionManager.deviceMotionAvailable else {
            print("accelerometer, gyro and device motion unavailable")
            return
        }
        
        self.motionManager.startAccelerometerUpdates()
        self.motionManager.startGyroUpdates()
        self.motionManager.startDeviceMotionUpdatesUsingReferenceFrame(CMAttitudeReferenceFrame.XTrueNorthZVertical)
        self.timer = NSTimer.scheduledTimerWithTimeInterval(0.1, target: self, selector: "timerTick", userInfo: nil, repeats: true)
    }

    override func viewWillDisappear(animated: Bool) {
        self.timer?.invalidate()
        self.timer = nil
    }
    
    func timerTick() {
        guard self.motionManager.deviceMotionAvailable else {
            return
        }
        
        guard self.motionManager.deviceMotion != nil else {
            return
        }
        
        let roll : Double = self.motionManager.deviceMotion!.attitude.roll
        let yaw : Double = self.motionManager.deviceMotion!.attitude.yaw
        let pitch : Double = self.motionManager.deviceMotion!.attitude.pitch
        let qw : Double = self.motionManager.deviceMotion!.attitude.quaternion.w
        let qx : Double = self.motionManager.deviceMotion!.attitude.quaternion.x
        let qy : Double = self.motionManager.deviceMotion!.attitude.quaternion.y
        let qz : Double = self.motionManager.deviceMotion!.attitude.quaternion.z
        
        let prediction = self.neuralNetwork.predict([roll, yaw, pitch, qw, qx, qy, qz])[0]
        if prediction >= 0.9 {
            self.view.backgroundColor = UIColor.blueColor()
        } else {
            self.view.backgroundColor = UIColor.orangeColor()
        }
        
        var predictionConfidence = prediction * 100
        if predictionConfidence > 100 { predictionConfidence = 100 }
        if predictionConfidence < 0 { predictionConfidence = 0 }
        self.orientationLabel.text = "\(String(format: "%.04f", predictionConfidence))% confident vertical"
    }

}