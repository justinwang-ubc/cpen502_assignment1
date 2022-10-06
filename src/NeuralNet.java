import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

// !task1: initialize weight for input-hidden weight 3*4 matrix including bias term
// !task: initialize weight for hidden to output weight length 5 array including a bias term
// task forward propagation:  output for each hidden neuron   array[4]
// task forward propagation: implement activation function for both bipolar and binary
// task forward propagation: output for output neuron: yi = f(sj)
// task backward propagation: compute error signal for output neuron: errorSignal = (Cj -yi) * f'(Sj)
// task backward propagation: compute error signal for hidden neuron: errorSignalHiddenNeuron =  yi *(1-yi)* (errorSignalOutput neuron * associated weight)
// task update weight for hidden-output weight
// task update weight for input-hidden layer weights
// check total error
public class NeuralNet implements NeuralNetInterface {
    // for encapsulation
    private int numInputs=2;
    private int numHidden=4;
    private int numOutput=1;
    private double learningRate=0.2;
    private double momentum=0;
    private int argA = 0;
    private int argB = 1;

    // forward propagation: output for hidden layer yi = f(sj)  sj are hidden layer input / output for output neuron
    private double[] hiddenNeuronsOutput = new double[numHidden + 1];
    private double[] outputNeuronOutput = new double[numOutput];
    private double[] inputVector = new double[numInputs +1];

    // weight for input to hidden neurons and weights for hidden neurons to output neurons
    private double[][] inputHiddenWeights = new double[numInputs + 1][numHidden];
    private double[][] hiddenOutputWeights = new double[numHidden + 1][numOutput];

    // error signal for hidden unit and output unit
    private double[] hiddenUnitsErrorSignal = new double[numHidden];
    private double[] outputUnitErrorSignal = new double[numOutput];

    //  store previous weight change into array so that can update weight using momentum term
    private double[][] inputHiddenWeightsChange = new double[numInputs + 1][numHidden];
    private double[][] hiddenOutputWeightsChange = new double[numHidden + 1][numOutput];

    private boolean trainingDataType;
    private int epoch;
    private double[] outputError = new double[numOutput];
    private double[] totalError = new double[numOutput];
    private double[][] trainingInputVector;
    private double[][] actualOutput;


    // add the total error into the linkList and then write into a file
    private List<String> totalErrorList = new LinkedList<>();

    // class constructor
    public NeuralNet(int numInputs, int numHidden, int numOutput, double learningRate, double momentum, int argA, int argB, boolean trainingDataType){
        this.numInputs = numInputs;
        this.numHidden = numHidden;
        this.numOutput = numOutput;
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.argA = argA;
        this.argB = argB;
        this.trainingDataType = trainingDataType;
    }


     // activation function for binary representation of inputVector
    @Override
    public double sigmoid(double x){
        return 1.0 / (1.0 + Math.exp(-x) ) ;
    }

    // activation function for bipolar representation of inputVector
    @Override
    public double customSigmoid(double x){
        return (2) / (1 + Math.exp(-x)) - 1 ;
    }

    // initialize weight matrix [-0.5,0.5]
    @Override
    public void initializeWeights(){
        for(int i = 0; i < inputHiddenWeights.length; i++){
            for(int j = 0; j < inputHiddenWeights[i].length; j++){
                inputHiddenWeights[i][j] = Math.random() - 0.5;
            }
        }
        for(int i = 0; i < hiddenOutputWeights.length; i++){
            for(int j = 0; j < hiddenOutputWeights[i].length; j++){
                hiddenOutputWeights[i][j] = Math.random() - 0.5;
            }
        }
    }

    @Override
    public void zeroWeights(){
    }

    public void InitializeTrainingSet(){
        if(trainingDataType){
            trainingInputVector = new double[][]{
                    {0, 0},
                    {0, 1},
                    {1, 0},
                    {1, 1}};
            actualOutput = new double[][]{
                    {0},
                    {1},
                    {1},
                    {0}};
        }else{
            trainingInputVector = new double[][]{
                    {-1, -1},
                    {-1, 1},
                    {1, -1},
                    {1, 1}};
            actualOutput = new double[][]{
                    {-1},
                    {1},
                    {1},
                    {-1}};
        }

    }

    // setter function
    public void SetMomentum(double x){
        this.momentum = x;
    }
    public void AddBiasTerm(double[] inputData){
        for(int i = 0; i < inputData.length; i++){
            this.inputVector[i] = inputData[i];
        }
        // bias term at the end of array
        this.inputVector[numInputs] = bias;
        this.hiddenNeuronsOutput[numHidden] = bias;
    }

    public void ForwardPropagation(double[] inputData){
        this.AddBiasTerm(inputData);
        for(int j = 0; j < numHidden; j++){
            for(int i = 0; i < numInputs + 1; i++){
                hiddenNeuronsOutput[j] += inputHiddenWeights[i][j] * inputVector[i];
            }

            if(trainingDataType){
                hiddenNeuronsOutput[j] = this.sigmoid(hiddenNeuronsOutput[j]);
            }else{
                hiddenNeuronsOutput[j] = this.customSigmoid(hiddenNeuronsOutput[j]);
            }
        }

        for(int j = 0; j < numOutput; j++){
            for(int i = 0; i < numHidden + 1; i++){
                outputNeuronOutput[j] += hiddenOutputWeights[i][j] * hiddenNeuronsOutput[i];
            }

            if(trainingDataType){
                outputNeuronOutput[j] = this.sigmoid(outputNeuronOutput[j]);
            }else{
                outputNeuronOutput[j] = this.customSigmoid(outputNeuronOutput[j]);
            }
        }
    }

    public void backwardPropagation(){
        // error signal for output unit
        for(int i = 0; i < numOutput; i++){
            if(trainingDataType){
                outputUnitErrorSignal[i] = outputNeuronOutput[i] * (1 - outputNeuronOutput[i]) * outputError[i];
            }else{
                outputUnitErrorSignal[i] = 0.5 * (1 - Math.pow(outputNeuronOutput[i],2)) * outputError[i];
            }
        }
        for(int i = 0; i < numOutput; i++){
            for(int j = 0; j < numHidden + 1; j++){
                hiddenOutputWeightsChange[j][i] = momentum * hiddenOutputWeightsChange[j][i] + learningRate * outputUnitErrorSignal[i] * hiddenNeuronsOutput[j];
                hiddenOutputWeights[j][i] += hiddenOutputWeightsChange[j][i];
            }
        }
        for(int i = 0; i < numHidden; i++){
            hiddenUnitsErrorSignal[i] = 0;
            for(int j = 0; j < numOutput; j++){
                hiddenUnitsErrorSignal[i] = hiddenUnitsErrorSignal[i] + hiddenOutputWeights[i][j] * outputUnitErrorSignal[j];
            }
            if(trainingDataType){
                hiddenUnitsErrorSignal[i] = hiddenNeuronsOutput[i] * (1 - hiddenNeuronsOutput[i]) * hiddenUnitsErrorSignal[i];
            }else{
                hiddenUnitsErrorSignal[i] = (0.5 * ( 1 - Math.pow(hiddenNeuronsOutput[i],2))) * hiddenUnitsErrorSignal[i];
            }
        }
        for(int i = 0; i < numHidden; i++){
            for(int j = 0; j < numInputs + 1; j++){
                inputHiddenWeightsChange[j][i] = momentum * inputHiddenWeightsChange[j][i] + learningRate * hiddenUnitsErrorSignal[i] * inputVector[j];
                inputHiddenWeights[j][i] += inputHiddenWeightsChange[j][i];
            }
        }
    }

    public int TrainNN(){
        this.totalErrorList.clear();
        this.epoch = 0;
        do{
            for(int i = 0; i < numOutput; i++){
                totalError[i] = 0;
            }
            for(int i = 0; i < trainingInputVector.length; i++){
                this.ForwardPropagation(trainingInputVector[i]);
                // i epoch for 4 training input
                for(int j = 0; j < numOutput; j++){
                    outputError[j] = actualOutput[i][j] - outputNeuronOutput[j];
                    totalError[j] += Math.pow(outputError[j],2);
                }
                this.backwardPropagation();
            }

            for(int i = 0; i < numOutput; i++){
                totalError[i] /= 2;
            }
            totalErrorList.add(String.valueOf(totalError[0]));
            epoch++;
        }while(totalError[0] > 0.05);
            return epoch;
    }

    public void CleanFile(){
        File file = new File("./Data");
        if(file.isDirectory()){
            String[] eachFilePath = file.list();
            if(eachFilePath != null){
                for(String path : eachFilePath){
                    File eachFile = new File(file.getAbsoluteFile() + "/" +path);
                    eachFile.delete();
                }
            }
        }
    }

    public void saveErrorVersusEpochs(){
        try{
            String dataType;
            if(trainingDataType){
                dataType = "Binary";
            }else{
                dataType = "Bipolar";
            }

            FileWriter fWriter = new FileWriter("./Data/ErrorVsEpochs-" + dataType + "-NumEpoch-"+epoch+".txt");
            for(String error: totalErrorList){
                fWriter.write(error + "\n");
            }
            fWriter.close();
        }catch (IOException e){
            System.out.println(e.getMessage());
        }
    }

    public void saveErrorVersusEpochsWithMomentum(){
        try{
            String dataType;
            if(trainingDataType){
                dataType = "Binary";
            }else{
                dataType = "Bipolar";
            }

            FileWriter fWriter = new FileWriter("./Data/ErrorVsEpochsWithMomentum-"+dataType+"-NumEpoch-"+epoch+".txt");
            for(String error : totalErrorList){
                fWriter.write(error + "\n");
            }
            fWriter.close();
        }catch (IOException e){
            System.out.println(e.getMessage());
        }
    }










}
