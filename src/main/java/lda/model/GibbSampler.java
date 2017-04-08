package lda.model;

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
	int[][] docFreqMatrix;

	double alpha;
	double beta = 0.01;

	double[][] phi; // Phi distribution: topic-word
	double[][] theta; // Theta distribution: doc-topic

	double Word_Beta;
	double Topic_Alpha;

	ArrayList<Pair>[] topicOfWordInDoc;
	int[] topicOfDoc;

	int[][] topic_word_matrix; // Frequency Matrix of topic-word
	int[][] doc_topic_matrix; // Frequency Matrix of doc-topic

	int[] doc_len; // Number of word in a document
	int[] topic_len; // Number of word in a topic

	double[] p; // Probability of topic K of current sampling word.

	public void initialize(int[][] docFreqMatrix, int num_topic) {
		V = docFreqMatrix.length;
		M = docFreqMatrix[0].length;
		K = num_topic;
		this.docFreqMatrix = new int[M][V];
		this.phi = new double[K][V];
		this.theta = new double[M][K];
		this.topicOfDoc = new int[M];
		this.topicOfWordInDoc = new ArrayList[M];
		this.topic_word_matrix = new int[K][V];
		this.doc_topic_matrix = new int[M][K];
		this.doc_len = new int[M];
		this.topic_len = new int[K];
		this.p = new double[K];
		this.docFreqMatrix = docFreqMatrix;
		alpha = 50.0 / K;
		Word_Beta = V * beta;
		Topic_Alpha = K * alpha;
		for (int m = 0; m < M; m++) {
			int topic = (int) Math.floor(Math.random() * K);
			topicOfDoc[m] = topic;
			topicOfWordInDoc[m] = new ArrayList<Pair>();
			for (int v = 0; v < V; v++) {
				for (int count = 0; count < docFreqMatrix[v][m]; count++) {
					topic = (int) Math.floor(Math.random() * K);
					Pair p = new Pair(topic, v); // Left: topic. Right: word
					topicOfWordInDoc[m].add(p);
					topic_len[topic]++;
					doc_len[m]++;
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
			p[k] = ((topic_word_matrix[k][word] + beta) / (topic_len[topic] + Word_Beta))
					* ((doc_topic_matrix[m][k] + alpha) / (doc_len[topic] + Topic_Alpha));
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

		topicOfWordInDoc[m].set(n, new Pair(topic, n));

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
		for (int iter = 0; iter < num_iteration; iter++) {
			for (int m = 0; m < M; m++) {
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

	public static void main(String[] args) {
		int[][] example = { { 1, 0, 1, 0, 0 }, // romeo
				{ 1, 1, 0, 0, 0 }, // juliet
				{ 0, 1, 0, 0, 0 }, // happy
				{ 0, 1, 1, 0, 0 }, // dagger
				{ 0, 0, 0, 1, 0 }, // live
				{ 0, 0, 1, 1, 0 }, // die
				{ 0, 0, 0, 1, 0 }, // free
				{ 0, 0, 0, 1, 1 }, // new-hamsphire
		};
		GibbSampler model = new GibbSampler();

		model.initialize(example, 2);
		model.estimate(1000);
		model.printModel();
		
		model.testQuery(0, 1);
		
		model.testQuery(3, 5);
	}

}
