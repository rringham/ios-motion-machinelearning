//
//  NeuralNetwork.swift
//  MotionMachineLearning
//
//  Created by Rob Ringham on 1/4/16.
//  Copyright © 2016 Ringham. All rights reserved.
//

import Accelerate
import Foundation

class NeuralNetwork {
    
    private let weightVectors: [[Double]]
    
    init(weightsFileName: String) {
        self.weightVectors = NeuralNetworkWeightFileProcessor.loadWeights(weightsFileName)
    }
    
    /*
    vDSP_mmulD( matrixA, 1, matrixB, 1, matrixAB, 1, X, Y, Z )
    
    the 1s should be left alone in most situations
    X - the number of rows in matrix A
    Y - the number of columns in matrix B
    Z - the number of columns in matrix A and the number of rows in matrix B.
    */
    
    func predict(X: [Double]) -> [Double] {
        let hiddenLayerSizes = [28,21,14,7]
        
        // X
        var Xbiased = [Double](X)
        Xbiased.insert(1.0, atIndex: 0)
        
        // input layer -> hidden layer 1
        // 7 input nodes -> 28 neurons
        let X_rows = 1
        let ϴ1 = self.weightVectors[0]
        let ϴ1_rows = ϴ1.count / hiddenLayerSizes[0]
        let ϴ1_cols = ϴ1.count / ϴ1_rows
        var a1 = [Double](count: X_rows * ϴ1_cols, repeatedValue: 0.0)
        vDSP_mmulD(Xbiased, 1, ϴ1, 1, &a1, 1, vDSP_Length(X_rows), vDSP_Length(ϴ1_cols), vDSP_Length(ϴ1_rows))
        var h1 = vSigmoid(a1)
        
        // hidden layer 1 -> hidden layer 2
        // 28 neurons -> 21 neurons
        h1.insert(1.0, atIndex: 0)
        let h1_rows = 1
        let ϴ2 = self.weightVectors[1]
        let ϴ2_rows = ϴ2.count / hiddenLayerSizes[1]
        let ϴ2_cols = ϴ2.count / ϴ2_rows
        var a2 = [Double](count: h1_rows * ϴ2_cols, repeatedValue: 0.0)
        vDSP_mmulD(h1, 1, ϴ2, 1, &a2, 1, vDSP_Length(h1_rows), vDSP_Length(ϴ2_cols), vDSP_Length(ϴ2_rows))
        var h2 = vSigmoid(a2)
        
        // hidden layer 2 -> hidden layer 3
        // 21 neurons -> 14 neurons
        h2.insert(1.0, atIndex: 0)
        let h2_rows = 1
        let ϴ3 = self.weightVectors[2]
        let ϴ3_rows = ϴ3.count / hiddenLayerSizes[2]
        let ϴ3_cols = ϴ3.count / ϴ3_rows
        var a3 = [Double](count: h2_rows * ϴ3_cols, repeatedValue: 0.0)
        vDSP_mmulD(h2, 1, ϴ3, 1, &a3, 1, vDSP_Length(h2_rows), vDSP_Length(ϴ3_cols), vDSP_Length(ϴ3_rows))
        var h3 = vSigmoid(a3)
        
        // hidden layer 3 -> hidden layer 4
        // 14 neurons -> 7 neurons
        h3.insert(1.0, atIndex: 0)
        let h3_rows = 1
        let ϴ4 = self.weightVectors[3]
        let ϴ4_rows = ϴ4.count / hiddenLayerSizes[3]
        let ϴ4_cols = ϴ4.count / ϴ4_rows
        var a4 = [Double](count: h3_rows * ϴ4_cols, repeatedValue: 0.0)
        vDSP_mmulD(h3, 1, ϴ4, 1, &a4, 1, vDSP_Length(h3_rows), vDSP_Length(ϴ4_cols), vDSP_Length(ϴ4_rows))
        var h4 = vSigmoid(a4)
        
        // hidden layer 4 -> output layer
        // 7 neurons -> 1 output node
        h4.insert(1.0, atIndex: 0)
        let h4_rows = 1
        let ϴ5 = self.weightVectors[4]
        let ϴ5_rows = ϴ5.count / 1
        let ϴ5_cols = ϴ5.count / ϴ5_rows
        var a5 = [Double](count: h4_rows * ϴ5_cols, repeatedValue: 0.0)
        vDSP_mmulD(h4, 1, ϴ5, 1, &a5, 1, vDSP_Length(h4_rows), vDSP_Length(ϴ5_cols), vDSP_Length(ϴ5_rows))
        let h5 = a5
        
        return h5
    }
    
    private func vSigmoid(vz: [Double]) -> [Double] {
        // formula: g = 1.0 ./ (1.0 + exp(-z))
        
        // compute (1.0 + exp(-z)
        var vexpZ_plus1 = [Double]()
        for z in vz {
            vexpZ_plus1.append(1.0 + exp(-z))
        }
        
        let vOnes = [Double](count: vz.count, repeatedValue: 1.0)
        var vg = [Double](count: vz.count, repeatedValue: 0.0)
        
        // compute 1.0 ./ vexpZ_plus1
        vDSP_vdivD(vexpZ_plus1, 1, vOnes, 1, &vg, 1, vDSP_Length(vg.count))
        
        return vg
    }
    
}