#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "people.h"

#define MAX_AGE 120

void printHist(int* hist);
void printPeople(person_t* people, int number);

int main(int argc, char** argv){
  
  if(argc != 3) {
    printf("Usage: list_gen [amount][any number]\nExample: ./list_gen 5 1229");
    return EXIT_FAILURE;
  }
  
  int amount_people= atoi(argv[1]);
  time_t startRand=atoi(argv[2]);
  
  person_t* people=(person_t*)malloc(amount_people*sizeof(person_t));
  person_t* people_sorted=(person_t*)malloc(amount_people*sizeof(person_t));
  int* count_sort_hist=(int*)malloc(MAX_AGE*sizeof(int));
  int* tmp_hist=(int*)malloc(MAX_AGE*sizeof(int));
  
  memset(count_sort_hist,0,sizeof(int)*MAX_AGE);
  
  srand(time(&startRand));
  int random=0;
  
  //generate people randomly
  for(int i=0; i<amount_people;i++){
    random=rand()%MAX_AGE;
    gen_name((people+i)->name);
    (people+i)->age=random;
  }
  
  printPeople(people,amount_people);
  
  /*sort*/
  printf("\n----------------------------\n");
  
  //calc histogram
  for(int i=0; i<MAX_AGE;i++){
      for(int j=0; j<amount_people; j++){
	  if((people+j)->age==i){
	      (*(count_sort_hist+i))++;
	  }
      }
  }

  //prefix sum
  tmp_hist[0]=0;
  for(int i=1; i<MAX_AGE; i++){
      tmp_hist[i]=count_sort_hist[i-1]+tmp_hist[i-1];
  }
  
  //last stage
  for(int i=0; i<amount_people; i++){
      int index_hist=(people+i)->age;
      int index_hist2=*(tmp_hist+index_hist);
      *(people_sorted+index_hist2)=*(people+i);
      (*(tmp_hist+index_hist))++;
  }
  
  printPeople(people_sorted,amount_people); 
  
  free(people);
  free(people_sorted);
  free(count_sort_hist);
  free(tmp_hist);
  
  return EXIT_SUCCESS; 
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