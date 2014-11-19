#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "people.h"

#define MAX_AGE 120

void countSortLastStage(person_t* people,person_t* people_sorted, int* tmp_hist, int entries);
void prefixSum(int* tmp, int* hist);
void calcHistogram(person_t* people, int* hist, int amount_people);
void generate_list(person_t* people, int number);
void printHist(int* hist);
void printPeople(person_t* people, int number);

int main(int argc, char** argv){
  
  if(argc != 3) {
    printf("Usage: list_gen [amount][any number]\nExample: ./list_gen 5 1229");
    return EXIT_FAILURE;
  }
  
  int amount_people= atoi(argv[1]);
  time_t startRand=atoi(argv[2]);
  srand(time(&startRand));
  
  person_t* people=(person_t*)malloc(amount_people*sizeof(person_t));
  person_t* people_sorted=(person_t*)malloc(amount_people*sizeof(person_t));
  int* count_sort_hist=(int*)malloc(MAX_AGE*sizeof(int));
  int* tmp_hist=(int*)malloc(MAX_AGE*sizeof(int));
  
  generate_list(people, amount_people);
  
  printPeople(people,amount_people);
  
  /*sort*/
  printf("\n----------------------------\n");
  
  calcHistogram(people, count_sort_hist,amount_people);

  prefixSum(tmp_hist, count_sort_hist);
  
  countSortLastStage(people, people_sorted, tmp_hist, amount_people);
  
  printPeople(people_sorted,amount_people); 
  
  //finalization
  free(people);
  free(people_sorted);
  free(count_sort_hist);
  free(tmp_hist);
  
  return EXIT_SUCCESS; 
}

void countSortLastStage(person_t* people,person_t* people_sorted, int* tmp_hist, int entries){
  
  //last stage
  for(int i=0; i<entries; i++){
      int index_hist=(people+i)->age;
      int index_hist2=*(tmp_hist+index_hist);
      *(people_sorted+index_hist2)=*(people+i);
      (*(tmp_hist+index_hist))++;
  }
}

void generate_list(person_t* people, int number){
  int random=0;
  //generate people randomly
  for(int i=0; i<number;i++){
    random=rand()%MAX_AGE;
    gen_name((people+i)->name);
    (people+i)->age=random;
  }
}

void prefixSum(int* tmp, int* hist){
    //prefix sum
  tmp[0]=0;
  for(int i=1; i<MAX_AGE; i++){
      tmp[i]=hist[i-1]+tmp[i-1];
  }
  
}

void calcHistogram(person_t* people, int* hist, int amount_people){
   //calc histogram
  for(int i=0; i<MAX_AGE;i++){
      for(int j=0; j<amount_people; j++){
	  if((people+j)->age==i){
	      (*(hist+i))++;
	  }
      }
  }
  
}

void printPeople(person_t* people, int number){
  for(int i=0; i<number; i++){
      printf("%d\t|\t%s\n",(people+i)->age, (people+i)->name);
  }
}

void printHist(int* hist){
 for(int i=0; i<MAX_AGE; i++){
      printf("Index[%d]:\t%d\t",i, *(hist+i));
      if((i+1)%5==0) printf("\n");
  }
}