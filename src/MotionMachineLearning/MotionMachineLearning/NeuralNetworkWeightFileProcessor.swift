//
//  NeuralNetworkWeightFileProcessor.swift
//  MotionMachineLearning
//
//  Created by Rob Ringham on 1/4/16.
//  Copyright © 2016 Ringham. All rights reserved.
//

import Foundation

class NeuralNetworkWeightFileProcessor {
    
    class func loadWeights(weightsFileName: String) -> [[Double]] {
        guard let weightRawStrings = rawWeightStringsFromContentsOfFileWithName(weightsFileName) else {
            print("could not load raw weight vector data from file \(weightsFileName).dat")
            return [[Double]]()
        }
        
        var weightVectors = [[Double]]()
        for weightRawString in weightRawStrings {
            var weights = [Double]()
            
            let rawValues = weightRawString.characters.split{$0 == " "}.map(String.init)
            for rawValue in rawValues {
                let weight = Double(rawValue)!
                weights.append(weight)
            }
            
            weightVectors.append(weights)
        }
        
        printLoadedWeightVectorsInfo(weightVectors)
        return weightVectors
    }
    
    class func printLoadedWeightVectorsInfo(weightVectors: [[Double]]) {
        print("loaded ϴ weight vectors:")
        
        var i = 1
        for weightVector in weightVectors {
            print("ϴ\(i++) weight vector length: \(weightVector.count)")
        }
        
        print("")
    }
    
    class func rawWeightStringsFromContentsOfFileWithName(fileName: String) -> [String]? {
        guard let path = NSBundle.mainBundle().pathForResource(fileName, ofType: "dat") else {
            return nil
        }
        
        do {
            let content = try String(contentsOfFile:path, encoding: NSUTF8StringEncoding)
            return content.componentsSeparatedByString("\n")
        } catch _ as NSError {
            return nil
        }
    }
    
}