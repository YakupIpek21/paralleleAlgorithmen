#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "people.h"

#define MAX_AGE 120

void printPeople(person_t* people, int number);

int main(int argc, char** argv){
  
  if(argc != 3) {
    printf("Usage: list_gen [amount][any number]\nExample: ./list_gen 5 1229");
    return EXIT_FAILURE;
  }
  
  int amount_people= atoi(argv[1]);
  time_t startRand=atoi(argv[2]);
  
  person_t* people=(person_t*)malloc(amount_people*sizeof(person_t));
  
  srand(time(&startRand));
  int random=0;
  
  //generate people randomly
  for(int i=0; i<amount_people;i++){
    random=rand()%MAX_AGE;
    gen_name((people+i)->name);
    (people+i)->age=random;
  }
  
  printPeople(people,amount_people);
  
  free(people);
  
  return EXIT_SUCCESS; 
}


void printPeople(person_t* people, int number){
  for(int i=0; i<number; i++){
      printf("%d\t|\t%s\n",(people+i)->age, (people+i)->name);
  }
}