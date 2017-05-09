package lda.model;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.*;

//class Pair {
//
//	private final Integer left;
//	private final Integer right;
//
//	public Pair(Integer left, Integer right) {
//		this.left = left;
//		this.right = right;
//	}
//
//	public int getLeft() {
//		return left;
//	}
//
//	public int getRight() {
//		return right;
//	}
//
//	@Override
//	public int hashCode() {
//		return left.hashCode() ^ right.hashCode();
//	}
//
//	@Override
//	public boolean equals(Object o) {
//		if (!(o instanceof Pair))
//			return false;
//		Pair pairo = (Pair) o;
//		return this.left.equals(pairo.getLeft())
//				&& this.right.equals(pairo.getRight());
//	}
//
//}

class Triple {
	
	private final Integer word;
	private final Byte x;
	private final Byte y;
	private final Integer z;

	public Triple(Integer word, Byte x, Byte y, Integer z) {
		this.word = word;
		this.x = x;
		this.y = y;
		this.z = z;
	}

	public byte getX() {
		return this.x;
	}

	public byte getY() {
		return this.y;
	}
	
	public int getZ() {
		return this.z;
	}
	
	public int getWord(){
		return this.word;
	}

	@Override
	public boolean equals(Object o) {
		if (!(o instanceof Triple))
			return false;
		Triple pairo = (Triple) o;
		return (this.word ==  pairo.getWord() && this.x == pairo.getX() && this.y == pairo.getY() && this.z == pairo.getZ());
	}

}

public class TopicSentimentMixture {

	int M; // Number of docs
	int V; // Number of words
	int K; // Number of topics
	
	double alpha;
	double beta = 0.01;
	double sigma = 2;
	double mu;

	double[][][] phi; // Phi distribution: topic-word
	double[][] theta; // Theta distribution: doc-topic
	double[] lambda;  // Lambda distribution: X coin
	double[][][] delta; // Delta distribution: Y coin

	double Word_Beta;
	double Topic_Alpha;
	double Two_Sigma;
	double Three_Mu;

	ArrayList<Triple>[] dataOfWordInDoc;

	int[][][] topic_y_word_matrix; // Frequency Matrix of topic-word
	int[][][] doc_y_topic_matrix; // Frequency Matrix of doc-topic
	int[][] doc_topic_matrix; // Frequency Matrix of doc-topic from Y
	
	byte[] xEqual1OfWord;
	int[] totalWordByX;
	int[][] totalWordInDocByX;
	int[][] totalWordInTopicByY;
	int totalWordInstance;
	

	double[] p; // Probability of topic K of current sampling word.
	
	public void initialize_new(ArrayList<ArrayList<Pair>> docFreqMatrix, int num_topic, int term_set_size) {
		V = term_set_size;
		M = docFreqMatrix.size();
		K = num_topic;
		this.xEqual1OfWord = new byte[V];
		this.totalWordInDocByX = new int[M][2];
		this.totalWordInTopicByY = new int[K][3];
		this.totalWordByX = new int[2];
		this.phi = new double[K][V][3];
		this.theta = new double[M][K];
		this.delta = new double[M][K][3];
		this.lambda = new double[2];
		this.dataOfWordInDoc = new ArrayList[M];
		this.topic_y_word_matrix = new int[K][3][V];
		this.doc_y_topic_matrix = new int[M][3][K];
		this.doc_topic_matrix = new int[M][K];
		this.p = new double[3 * K + 1];
		this.alpha = 50.0 / K;
		this.mu = this.alpha;
		this.Word_Beta = V * beta;
		this.Topic_Alpha = K * alpha;
		this.Two_Sigma = 2 * sigma;
		this.Three_Mu = 3 * mu;

		for (int m = 0; m < M; m++) {
			int z ;
			byte x, y;
			dataOfWordInDoc[m] = new ArrayList<Triple>();
			for (int word_in_doc = 0; word_in_doc < docFreqMatrix.get(m).size(); word_in_doc++) {
				Pair word_frequency = docFreqMatrix.get(m).get(word_in_doc);
				for (int count = 0; count < word_frequency.getRight(); count++) {   //Left: word; Right: Frequency
					x = (byte) Math.floor(Math.random() * 2);
					y = (byte) Math.floor(Math.random() * 3);
					z = (int) Math.floor(Math.random() * K);
					Triple data = new Triple(word_frequency.getLeft(), x, y, z); // Left: word. Right: data
					dataOfWordInDoc[m].add(data);
				}
			}
		}
		buildMatrices();
	}	

	public void buildMatrices() {
		for (int m = 0; m < M; m++) {
			for (int k = 0; k < dataOfWordInDoc[m].size(); k++) {
				byte x = dataOfWordInDoc[m].get(k).getX();
				byte y = dataOfWordInDoc[m].get(k).getY();
				int z = dataOfWordInDoc[m].get(k).getZ();
				int word = dataOfWordInDoc[m].get(k).getWord();
				totalWordByX[x]++;
				totalWordInDocByX[m][x]++;
				totalWordInstance ++;
				if (x == 1) 
				{
					xEqual1OfWord[word]++;
				}
				else 
				{
					topic_y_word_matrix[z][y][word]++;
					totalWordInTopicByY[z][y]++;
					doc_y_topic_matrix[m][y][z]++;
					doc_topic_matrix[m][z]++;
				}
				
			}
		}
		System.out.println("Matrix building is finished");
	}

	private void sampling(int m, int n) { 
		// compute on topic_y_word_matrix and
		// doc_y_topic_matrix variables
		int word = dataOfWordInDoc[m].get(n).getWord();
		byte x = dataOfWordInDoc[m].get(n).getX();
		byte y = dataOfWordInDoc[m].get(n).getY();
		int z = dataOfWordInDoc[m].get(n).getZ();
		
		totalWordByX[x] -= 1;
		totalWordInDocByX[m][x] -= 1;
		totalWordInstance -= 1;
		
		if (x  == 1) {
			xEqual1OfWord[word] -= 1;
			p[0] = ((totalWordByX[x] + sigma) / (totalWordInstance + Two_Sigma))
				* ((xEqual1OfWord[word] + beta) / (totalWordByX[x] + Word_Beta));
			xEqual1OfWord[word] += 1;
		}
		else {
			topic_y_word_matrix[z][y][word] -= 1;
			doc_y_topic_matrix[m][y][z] -= 1;
			totalWordInTopicByY[z][y] -= 1;
			for (int k = 0; k < K; k++) {
				
				// Compute p[] three times is faster than using an inner loop for y 
				
				p[3 * k + 1] = ((totalWordByX[x] + sigma) / (totalWordInstance + Two_Sigma))
						* ((doc_topic_matrix[m][k] + alpha) / (totalWordInDocByX[m][x] + Topic_Alpha))
						* ((doc_y_topic_matrix[m][0][k] + mu) / (totalWordInDocByX[m][x] + Three_Mu))
						* ((topic_y_word_matrix[k][0][word] + beta) / (totalWordInTopicByY[k][0] + Word_Beta));
	
				p[3 * k + 2] = ((totalWordByX[x] + sigma) / (totalWordInstance + Two_Sigma))
						* ((doc_topic_matrix[m][k] + alpha) / (totalWordInDocByX[m][x] + Topic_Alpha))
						* ((doc_y_topic_matrix[m][1][k] + mu) / (totalWordInDocByX[m][x] + Three_Mu))
						* ((topic_y_word_matrix[k][1][word] + beta) / (totalWordInTopicByY[k][1] + Word_Beta));
	
				p[3 * k + 3] = ((totalWordByX[x] + sigma) / (totalWordInstance + Two_Sigma))
						* ((doc_topic_matrix[m][k] + alpha) / (totalWordInDocByX[m][x] + Topic_Alpha))
						* ((doc_y_topic_matrix[m][2][k] + mu) / (totalWordInDocByX[m][x] + Three_Mu))
						* ((topic_y_word_matrix[k][2][word] + beta) / (totalWordInTopicByY[k][2] + Word_Beta));
				
				topic_y_word_matrix[z][y][word] += 1;
				doc_y_topic_matrix[m][y][z] += 1;
				totalWordInTopicByY[z][y] += 1;
			}
		}
		
		totalWordByX[x] += 1;
		totalWordInDocByX[m][x] += 1;
		totalWordInstance += 1;

		for (int k = 1; k < 3 * K + 1; k++) {
			p[k] += p[k - 1];
		}

		double threshold = Math.random() * p[3 *  K];
		
		for (int i = 0; i < 3 * K + 1; i++) {
			if (p[i] > threshold) {
				if (i == 0) 
					x = 0;
				else{
					x = 1;
					y = (byte)((i  - 1) % 3);
					z = (i  - 1) / 3;
				}
				break;
			}
		}

		dataOfWordInDoc[m].set(n, new Triple(word, x, y, z));  

	}

	public void computeTheta() {
		for (int m = 0; m < M; m++) {
			for (int k = 0; k < K; k++) {
				theta[m][k] = ((doc_topic_matrix[m][k] + alpha) / (totalWordInDocByX[m][0] + Topic_Alpha));
			}
		}
	}
	
	public void computeDelta() {
		for (int m = 0; m < M; m++) {
			for (int k = 0; k < K; k++) {
				for (int y = 0; y < 3; y++) {
					delta[m][k][y] = ((doc_y_topic_matrix[m][y][k] + mu) / (totalWordInDocByX[m][0] + Three_Mu));
				}
			}
		}
	}	

	public void computePhi() {

		for (int k = 0; k < K; k++) {
			for (int v = 0; v < V; v++) {
				for (byte y = 0; y < 3; y++) {
					phi[k][v][y] = ((topic_y_word_matrix[k][y][v] + beta) / (totalWordInTopicByY[k][y] + Word_Beta));
				}
			}
		}
	}
	
	public void computeLambda() {
		for (int x = 0; x < 2; x++) {
			lambda[x] = ((totalWordByX[x] + sigma) / (totalWordInstance + Two_Sigma));
		}
	}

	public void estimate(int num_iteration) {
		for (int iter = 0; iter < num_iteration; iter++) {
			for (int m = 0; m < M; m++) {
				for (int n = 0; n < dataOfWordInDoc[m].size(); n++) {
					sampling(m, n);
				}
			}
		}
		computePhi();
		computeTheta();
		computeDelta();
		computeLambda();
	}

	void printModel() {
		System.out.println("Theta");
		for (int m = 0; m < M; m++) {
			for (int n = 0; n < K; n++) {
				System.out.print(theta[m][n] + " ");
			}
			System.out.println();
		}

		System.out.println("Phi");
		for (int k = 0; k < K; k++) {
			System.out.println("Topic " + k);
			for (int v = 0; v < V; v++) {
				System.out.println("\t" + phi[k][v][0] + " " + phi[k][v][1] + " " + phi[k][v][2]);
			}
			System.out.println();
		}
		
		for (int m = 0; m < M; m++) {
			System.out.println("Document " + m);
			for (int k = 0; k < K; k++) {
				System.out.println("\t" + delta[m][k][0] + " " + delta[m][k][1] + " " + delta[m][k][2]);
			}
		}
		
		System.out.println("Lamda: p(x=0)=" + lambda[0] + "\tp(x=1)=" + lambda[1]);
		

	}
	
	public void writeTheta(String file){
		try {
			BufferedWriter bw = new BufferedWriter(
				new FileWriter(new File(file)));
			for (int m = 0; m < M; m++) {
				for (int n = 0; n < K; n++) {
					bw.write(theta[m][n] + " ");
				}
				bw.write("\r\n");
			}
			bw.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public void writePhi(String file){
		try {
			BufferedWriter bw = new BufferedWriter(
				new FileWriter(new File(file)));
			for (int k = 0; k < K; k++) {
				bw.write("\r\n");
				for (int v = 0; v < V; v++) {
					bw.write(phi[k][v][0] + " " + phi[k][v][1] + " " + phi[k][v][2] + "\r\n");
				}
				bw.write("\r\n");
			}
			bw.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public void writeDelta(String file){
		try {
			BufferedWriter bw = new BufferedWriter(
				new FileWriter(new File(file)));
			for (int m = 0; m < M; m++) {
				bw.write("\r\n");
				for (int k = 0; k < K; k++) {
					bw.write(delta[m][k][0] + " " + delta[m][k][1] + " " + delta[m][k][2] + "\r\n");
				}
			}
			bw.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public void writeLambda(String file){
		try {
			BufferedWriter bw = new BufferedWriter(
				new FileWriter(new File(file)));
			bw.write(lambda[0]  + "\t" + lambda[1]);
			bw.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	
	public void writeModels(String path) {
		writeTheta(path + "\\theta.models");
		writePhi(path + "\\phi.models");
		writeDelta(path + "\\delta.models");
		writeLambda(path + "\\lambda.models");
	}
	
	public ArrayList<ArrayList<Pair>> prepare_data() {
//		String doc0 = "Romeo and Juliet";
//		String doc1 = "Juliet: O happy dagger";
//		String doc2 = "Romeo died by dagger";
//		String doc3 = "\"Live free or die\", that's the new-Hampshire's motto";
//		String doc4 = "Did you know, New-Hampsphire is in New-England";
//		
//		int[][] example = { 
//				{ 1, 0, 1, 0, 0 }, 				// romeo
//				{ 1, 1, 0, 0, 0 }, 				// juliet
//				{ 0, 1, 0, 0, 0 }, 				// happy
//				{ 0, 1, 1, 0, 0 }, 				// dagger
//				{ 0, 0, 0, 1, 0 }, 				// live
//				{ 0, 0, 1, 1, 0 }, 				// die
//				{ 0, 0, 0, 1, 0 }, 				// free
//				{ 0, 0, 0, 1, 1 },				// new-hamsphire
//		};
		
		//Pair<Left, Right>: Left is word. Right´is frequency
		
		ArrayList<ArrayList<Pair>> doc_vector = new ArrayList<ArrayList<Pair>>();
		ArrayList<Pair> doc0 = new ArrayList<Pair>();
		doc0.add(new Pair(0, 1));
		doc0.add(new Pair(2, 1));  
		ArrayList<Pair> doc1 = new ArrayList<Pair>();
		doc1.add(new Pair(1, 1));
		doc1.add(new Pair(2, 1));
		doc1.add(new Pair(3, 1));  
		ArrayList<Pair> doc2 = new ArrayList<Pair>();
		doc2.add(new Pair(0, 1));
		doc2.add(new Pair(3, 1));
		doc2.add(new Pair(5, 1));  
		ArrayList<Pair> doc3 = new ArrayList<Pair>();
		doc3.add(new Pair(4, 1));
		doc3.add(new Pair(5, 1));
		doc3.add(new Pair(6, 1));
		doc3.add(new Pair(7, 1));  
		ArrayList<Pair> doc4 = new ArrayList<Pair>();
		doc4.add(new Pair(1,7));  
		doc_vector.add(doc0);
		doc_vector.add(doc1);
		doc_vector.add(doc2);
		doc_vector.add(doc3);
		doc_vector.add(doc4);
		return doc_vector;
	}

	public static void main(String[] args) {

		
		
		TopicSentimentMixture model = new TopicSentimentMixture();
//		Change you path bellow
		LoadDocsFromFile load_data = new LoadDocsFromFile("Bitterlemons.docs", "Bitterlemons.words");
		load_data.load();
		model.initialize_new(load_data.getDocs(), 200, 4451);
//		or
//		model.initialize_new(model.prepare_data(), 2, 8); // This line using sample data
		
		model.estimate(100);
//		Change you path bellow
		model.writeModels("Path");
//		model.printModel();
	}

}
