// Fill out your copyright notice in the Description page of Project Settings.


#include "DuckiebotAgentController.h"


// Sets default values
ADuckiebotAgentController::ADuckiebotAgentController()
{
	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;
	UE_LOG(LogTemp, Warning, TEXT("DuckiebotAgent Controller Initialized"));
}

// Called when the game starts or when spawned
void ADuckiebotAgentController::BeginPlay()
{
	Super::BeginPlay();
	
}

// Called every frame
void ADuckiebotAgentController::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
}

