package lda.model;

import java.io.*;
import java.util.*;

public class LoadDocsFromFile {
	String doc_file;
	String dict_file;
	ArrayList<ArrayList<Pair>> docs;
	
	String[] dict;
	public LoadDocsFromFile(String docfile, String dictionary) {
		doc_file = docfile;
		dict_file = dictionary;
	}
	public void load(){
		try {
			BufferedReader br = new BufferedReader(new FileReader(new File(doc_file)));
			String line_of_doc;
			String[] components;
			String[] word_data;
			ArrayList<Pair> doc_content ;
			docs = new ArrayList<ArrayList<Pair>>();

			while ((line_of_doc = br.readLine()) != null) {
				components = line_of_doc.split(" ");
				doc_content = new ArrayList<Pair>();
				for (int i = 1; i < components.length; i ++) {
					word_data = components[i].split(":", 2);
					doc_content.add(new Pair(Integer.parseInt(word_data[0]), Integer.parseInt(word_data[1])));
				}
				docs.add(doc_content);
			}
			br.close();
			
			System.out.println("Finish loading docs...");
			
			br = new BufferedReader(new FileReader(new File(dict_file)));
			int num_of_line = 0;
			while (br.readLine() != null) {
				num_of_line++;
			}
			
			dict = new String[num_of_line];
			br = new BufferedReader(new FileReader(new File(dict_file)));
			for(int i = 0; i < num_of_line; i ++) {
				line_of_doc = br.readLine();
				components = line_of_doc.split(" ", 2);
				if (components.length == 2) {
					dict[Integer.parseInt(components[0])] = components[1];
				}
				else {
					dict[i] = components[0];
				}	
			}
			br.close();
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public ArrayList<ArrayList<Pair>> getDocs(){
		return docs;
	}
	
	public String[] getWordDict(){
		return dict;
	}
}
