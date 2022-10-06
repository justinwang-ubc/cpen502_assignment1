public class Main {


    public static void main(String[] args){
        // a)
        boolean isBinary = true;
        int TotalEpochs = 0;
        int numOfTraining = 200;
        NeuralNet XorBinary = new NeuralNet(2, 4,1,0.2,0,0,1,isBinary);
        XorBinary.InitializeTrainingSet();
        XorBinary.CleanFile();
        for(int i =0; i <= numOfTraining; i++){
            XorBinary.initializeWeights();
            TotalEpochs += XorBinary.TrainNN();
            XorBinary.saveErrorVersusEpochs();
        }
        int averageEpochs =  TotalEpochs / numOfTraining;
        System.out.println("Average epochs for binary take to reach a total error of less than 0.05 are: " + averageEpochs);

        // b)
        isBinary = false;
        TotalEpochs = 0;
        NeuralNet XorBipolar = new NeuralNet(2,4,1,0.2,0, -1,1, isBinary);
        XorBipolar.InitializeTrainingSet();
        for(int i = 0; i < numOfTraining; i++){
            XorBipolar.initializeWeights();
            TotalEpochs += XorBipolar.TrainNN();
            XorBipolar.saveErrorVersusEpochs();
        }
        averageEpochs = TotalEpochs / numOfTraining;
        System.out.println("Average epochs for bipolar take to reach a total error of less than 0.05 are: " + averageEpochs);


        // c)  binary momentum = 0.9
        isBinary = true;
        TotalEpochs = 0;
        NeuralNet XorBinaryWithMomentum = new NeuralNet(2,4,1,0.2,0.9,0,1,isBinary);
        XorBinaryWithMomentum.InitializeTrainingSet();
        for(int i =0; i <= numOfTraining; i++){
            XorBinaryWithMomentum.initializeWeights();
            TotalEpochs += XorBinaryWithMomentum.TrainNN();
            XorBinaryWithMomentum.saveErrorVersusEpochsWithMomentum();
        }
         averageEpochs =  TotalEpochs / numOfTraining;
        System.out.println("With momentum = 0.9, Average epochs for binary take to reach a total error of less than 0.05 are: " + averageEpochs);


        //c) bipolar momentum = 0.9
        isBinary = false;
        TotalEpochs = 0;
        NeuralNet XorBipolarWithMomentum = new NeuralNet(2,4,1,0.2,0.9, -1,1, isBinary);
        XorBipolarWithMomentum.InitializeTrainingSet();
        for(int i = 0; i < numOfTraining; i++){
            XorBipolarWithMomentum.initializeWeights();
            TotalEpochs += XorBipolarWithMomentum.TrainNN();
            XorBipolarWithMomentum.saveErrorVersusEpochsWithMomentum();
        }
        averageEpochs = TotalEpochs / numOfTraining;
        System.out.println("With momentum = 0.9, Average epochs for bipolar take to reach a total error of less than 0.05 are: " + averageEpochs);



    }
}
