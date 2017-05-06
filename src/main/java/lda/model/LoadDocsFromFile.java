package lda.model;

import java.io.*;
import java.util.*;

public class LoadDocsFromFile {
	String doc_file;
	String dict_file;
	ArrayList<ArrayList<Pair>> docs;
	
	ArrayList<HashMap<Integer, String>> dict;
	public LoadDocsFromFile(String docfile, String dictionary) {
		doc_file = docfile;
		dict_file = dictionary;
		dict  = new ArrayList<HashMap<Integer,String>>();
	}
	public void load(){
		try {
			BufferedReader br = new BufferedReader(new FileReader(new File(doc_file)));
			String line_of_doc;
			String[] components;
			String[] word_data;
			ArrayList<Pair> doc_content = new ArrayList<Pair>();
			docs = new ArrayList<ArrayList<Pair>>();

			while ((line_of_doc = br.readLine()) != null) {
				components = line_of_doc.split(" ");
				for (int i = 1; i < components.length; i ++) {
					word_data = components[i].split(":", 2);
					doc_content.add(new Pair(Integer.parseInt(word_data[0]), Integer.parseInt(word_data[1])));
				}
				docs.add(doc_content);
				doc_content.clear();
			}
			
			System.out.println("Finish loading...");
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public ArrayList<ArrayList<Pair>> getDocs(){
		return docs;
	}
}
