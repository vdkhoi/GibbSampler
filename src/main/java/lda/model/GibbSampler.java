package lda.model;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.*;

//This code base on "Parameter Estimation for Text Analysis"
//And "Comparing Latent Dirichlet Allocation And Latent Semantic Analysis As Classifiers" dissertation

class Pair {

	private final Integer left;
	private final Integer right;

	public Pair(Integer left, Integer right) {
		this.left = left;
		this.right = right;
	}

	public int getLeft() {
		return left;
	}

	public int getRight() {
		return right;
	}

	@Override
	public int hashCode() {
		return left.hashCode() ^ right.hashCode();
	}

	@Override
	public boolean equals(Object o) {
		if (!(o instanceof Pair))
			return false;
		Pair pairo = (Pair) o;
		return this.left.equals(pairo.getLeft())
				&& this.right.equals(pairo.getRight());
	}

}

public class GibbSampler {

	int M; // Number of docs
	int V; // Number of words
	int K; // Number of topics
	
	String[] wordDict;
	
	double alpha;
	double beta = 0.01;

	double[][] phi; // Phi distribution: topic-word
	double[][] theta; // Theta distribution: doc-topic

	double Word_Beta;
	double Topic_Alpha;

	ArrayList<Pair>[] topicOfWordInDoc;


	int[][] topic_word_matrix; // Frequency Matrix of topic-word
	int[][] doc_topic_matrix; // Frequency Matrix of doc-topic

	int[] doc_len; // Number of word in a document
	int[] topic_len; // Number of word in a topic

	double[] p; // Probability of topic K of current sampling word.

	public void initialize(int[][] docFreqMatrix, int num_topic) {
		V = docFreqMatrix.length;
		M = docFreqMatrix[0].length;
		K = num_topic;
		this.phi = new double[K][V];
		this.theta = new double[M][K];
		this.topicOfWordInDoc = new ArrayList[M];
		this.topic_word_matrix = new int[K][V];
		this.doc_topic_matrix = new int[M][K];
		this.doc_len = new int[M];
		this.topic_len = new int[K];
		this.p = new double[K];
		alpha = 50.0 / K;
		Word_Beta = V * beta;
		Topic_Alpha = K * alpha;
		for (int m = 0; m < M; m++) {
			int topic;
			topicOfWordInDoc[m] = new ArrayList<Pair>();
			for (int v = 0; v < V; v++) {
				for (int count = 0; count < docFreqMatrix[v][m]; count++) {
					topic = (int) Math.floor(Math.random() * K);
					Pair p = new Pair(topic, v); // Left: topic. Right: word
					topicOfWordInDoc[m].add(p);
				}
			}
		}

		buildMatrices();
	}
	
	public void initialize_new(ArrayList<ArrayList<Pair>> docFreqMatrix, int num_topic, String[] word_list) {
		V = word_list.length;
		M = docFreqMatrix.size();
		K = num_topic;
		this.wordDict = word_list;
		this.phi = new double[K][V];
		this.theta = new double[M][K];
		this.topicOfWordInDoc = new ArrayList[M];
		this.topic_word_matrix = new int[K][V];
		this.doc_topic_matrix = new int[M][K];
		this.doc_len = new int[M];
		this.topic_len = new int[K];
		this.p = new double[K];
		alpha = 50.0 / K;
		Word_Beta = V * beta;
		Topic_Alpha = K * alpha;

		for (int m = 0; m < M; m++) {
			int topic;
			topicOfWordInDoc[m] = new ArrayList<Pair>(docFreqMatrix.get(m).size());
			for (int word_in_doc = 0; word_in_doc < docFreqMatrix.get(m).size(); word_in_doc++) {
				Pair word_frequency_pair = docFreqMatrix.get(m).get(word_in_doc);
				for (int count = 0; count < word_frequency_pair.getLeft(); count++) {   //Left: frequency; Right: word
					topic = (int) Math.floor(Math.random() * K);
					Pair p = new Pair(topic, word_frequency_pair.getRight()); // Left: topic. Right: word
					topicOfWordInDoc[m].add(p);
				}
			}
		}

		buildMatrices();
	}
	

	public void buildMatrices() {
		for (int m = 0; m < M; m++) {
			for (int k = 0; k < topicOfWordInDoc[m].size(); k++) {
				int topic = topicOfWordInDoc[m].get(k).getLeft();
				int word = topicOfWordInDoc[m].get(k).getRight();
				topic_word_matrix[topic][word]++;
				doc_topic_matrix[m][topic]++;
				topic_len[topic]++;
				doc_len[m]++;
			}
		}
	}

	private void sampling(int m, int n) { // compute on topic_word_matrix and
											// doc_topic_matrix variables

		int topic = topicOfWordInDoc[m].get(n).getLeft();
		int word = topicOfWordInDoc[m].get(n).getRight();

		topic_word_matrix[topic][word] -= 1;
		doc_topic_matrix[m][topic] -= 1;
		topic_len[topic] -= 1;
		doc_len[m] -= 1;

		for (int k = 0; k < K; k++) {
			p[k] = ((topic_word_matrix[k][word] + beta) / (topic_len[k] + Word_Beta))
					* ((doc_topic_matrix[m][k] + alpha) / (doc_len[m] + Topic_Alpha));
		}

		for (int k = 1; k < K; k++) {
			p[k] += p[k - 1];
		}
		
		double threshold = Math.random() * p[K - 1];

		for (topic = 0; topic < K - 1; topic++) {
			if (p[topic] > threshold)
				break;
		}

		topic_word_matrix[topic][word] += 1;
		doc_topic_matrix[m][topic] += 1;
		topic_len[topic] += 1;
		doc_len[m] += 1;

		topicOfWordInDoc[m].set(n, new Pair(topic, word));

	}

	public void computeTheta() {
		for (int m = 0; m < M; m++) {
			for (int k = 0; k < K; k++) {
				theta[m][k] = ((doc_topic_matrix[m][k] + alpha) / (doc_len[m] + Topic_Alpha));
			}
		}
	}

	public void computePhi() {
		for (int k = 0; k < K; k++) {
			for (int v = 0; v < V; v++) {
				phi[k][v] = ((topic_word_matrix[k][v] + beta) / (topic_len[k] + Word_Beta));
			}
		}
	}

	public void estimate(int num_iteration) {
		for (int iter = 0; iter < num_iteration; iter++) {  // Run iteration
			for (int m = 0; m < M; m++) {                   // Traverse on M documents
				for (int n = 0; n < topicOfWordInDoc[m].size(); n++) {  
					sampling(m, n);
				}
			}
		}
		computePhi();
		computeTheta();
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
			for (int v = 0; v < V; v++) {
				System.out.print(phi[k][v] + " ");
			}
			System.out.println();
		}

	}
	
	private int findMinIndex(int[] index_list, int k){
		double min = phi[k][index_list[0]];
		int index = 0;
		for (int i = 1; i < index_list.length; i++){
			if (phi[k][index_list[i]] < min) {
				min = phi[k][index_list[i]];
				index = i;
			}
		}
		return index;
	}
	
	public void writeTopWord(String file, int num) {
		try {
			String[][] top_words = new String[num][K];
			for (int k = 0; k < K; k++) {
				int[] index_of_num = new int[num];
				for (int v = 0; v < num; v++) {
					index_of_num[v] = v;
				}

				for (int v = num; v < V; v++) {
					int min_idx = findMinIndex(index_of_num, k);
					if (phi[k][v] > phi[k][index_of_num[min_idx]]) {
						index_of_num[min_idx] = v;
					}
				}

				// System.out.println("Topic k = " + k + " and y = " + y);
				for (int v = 19; v > -1; v--) {
					int min_idx = findMinIndex(index_of_num, k);
					top_words[v][k] = wordDict[index_of_num[min_idx]] + "("
							+ index_of_num[min_idx] + "-"
							+ phi[k][index_of_num[min_idx]] + ")";
					// System.out.println(top_words[v][k][y]);
					phi[k][index_of_num[min_idx]] = 10000000000.0;
				}

			}

			BufferedWriter bw = new BufferedWriter(new FileWriter(
					new File(file)));
			for (int v = 0; v < num; v++) {
				for (int k = 0; k < K; k++) {

					bw.write(top_words[v][k] + "\t");

				}
				bw.write("\r\n");
			}
			bw.close();

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	
	public void testQuery(int idx1, int idx2){
		double[] query = new double[K];
		for (int idx = 0; idx < K; idx ++)
			query[idx] = (phi[idx][idx1] + phi[idx][idx2]) * 0.5;
		double[] score = new double[M];
		for (int m = 0; m < M; m++ ){
			for (int k = 0; k < K; k++ ){
				score[m] += theta[m][k] * query[k];
			}
			System.out.println("Score with d" + m + " is: " + score[m]);
		}
	}
	
	public ArrayList<Pair>[] prepare_data() {
		ArrayList<Pair>[] doc_vector = new ArrayList[5];
		//Pair<Left, Right>: Left is frequency. Right´is word
		Pair[] doc0 = {new Pair(1,0), new Pair(1,2)};  					
		Pair[] doc1 = {new Pair(1,1), new Pair(1,2), new Pair(1,3)};  	
		Pair[] doc2 = {new Pair(1,0), new Pair(1,3), new Pair(1,5)};                 
		Pair[] doc3 = {new Pair(1,4), new Pair(1,5), new Pair(1,6), new Pair(1,7)};  
		Pair[] doc4 = {new Pair(1,7)};

		doc_vector[0] = new ArrayList<Pair>(Arrays.asList(doc0));
		doc_vector[1] = new ArrayList<Pair>(Arrays.asList(doc1));
		doc_vector[2] = new ArrayList<Pair>(Arrays.asList(doc2));
		doc_vector[3] = new ArrayList<Pair>(Arrays.asList(doc3));
		doc_vector[4] = new ArrayList<Pair>(Arrays.asList(doc4));
		
		return doc_vector;
	}

	public static void main(String[] args) {
		int[][] example = { 
				{ 1, 0, 1, 0, 0 }, 				// romeo
				{ 1, 1, 0, 0, 0 }, 				// juliet
				{ 0, 1, 0, 0, 0 }, 				// happy
				{ 0, 1, 1, 0, 0 }, 				// dagger
				{ 0, 0, 0, 1, 0 }, 				// live
				{ 0, 0, 1, 1, 0 }, 				// die
				{ 0, 0, 0, 1, 0 }, 				// free
				{ 0, 0, 0, 1, 1 }, 				// new-hamsphire
		};
		
		
		GibbSampler model = new GibbSampler();
		
		
//		model.initialize_new(model.prepare_data(), 2, 8);
//		or
//		model.initialize_new(example, 2, 8);
		
		
		LoadDocsFromFile load_data = new LoadDocsFromFile("E:\\OneDrive\\With_TAnh\\collaborative_work\\data\\ap.docs", "E:\\OneDrive\\With_TAnh\\collaborative_work\\data\\ap.words");
		load_data.load();
		model.initialize_new(load_data.getDocs(), 100, load_data.getWordDict());
		
		model.estimate(100);
		model.writeTopWord("E:\\OneDrive\\With_TAnh\\models\\LDA\\top_words.csv", 20);
		model.printModel();
		
		
	}

}
