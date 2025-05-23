// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "HolodeckPawnController.h"
#include "DuckiebotAgentController.generated.h"

UCLASS()
class DUCKIEBOTSSIM_API ADuckiebotAgentController : public AHolodeckPawnController
{
	GENERATED_BODY()

public:
	// Sets default values for this actor's properties
	ADuckiebotAgentController();

	void AddControlSchemes() override {
		// No control schemes
	}
	
protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:
	// Called every frame
	virtual void Tick(float DeltaTime) override;
};
