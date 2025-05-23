// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "HolodeckAgent.h"
#include "Containers/CircularQueue.h"
#include "DuckiebotAgent.generated.h"

static float MAX_DUCKIEBOT_ROTATION_SPEED = 1.0;
static float MAX_DUCKIEBOT_FORWARD_SPEED = 1.0;

UCLASS()
class DUCKIEBOTSSIM_API ADuckiebotAgent : public AHolodeckAgent
{
	GENERATED_BODY()

public:
	// Sets default values for this pawn's properties
	ADuckiebotAgent();

	void InitializeAgent() override;

	// Called every frame
	virtual void Tick(float DeltaTime) override;
	
	unsigned int GetRawActionSizeInBytes() const override { return 8 * sizeof(float); }; // VelCmd, TurnCmd, ResetCmd, ResetX, ResetY, ResetYaw, ResetForwardVelocity, ResetYawVelocity
	void* GetRawActionBuffer() const override { return (void*)CommandArray; };

	float GetVelocityCommand();
	float GetRotationCommand();
	bool GetResetCommand();
	FVector3f GetResetXYYawCommand();
	float GetResetForwardVelCommand();
	float GetResetYawVelCommand();

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float ProgressAlongIntendedPath;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float VelocityAlongIntendedPath;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float DistanceFromIntendedPath;
	
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	bool CollidedWithWhiteLine;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	bool CollidedWithYellowLine;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	bool IsYellowLineToLeftAndWhiteLineToRight;
	
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	bool EnteredNewRoadTile;

	UFUNCTION(BlueprintImplementableEvent)
	void UpdateDuckiebotLoopStatusBools();

	UFUNCTION(BlueprintImplementableEvent)
	void ClearTemporaryDuckiebotLoopStatusBools();

	UFUNCTION(BlueprintCallable)
	void ClearRewardTrackingState();

	UFUNCTION(BlueprintImplementableEvent)
	void ResetDuckiebotToNewLocation(FVector3f NewXYYaw);

	void ConsumeResetCommand();
	
protected:
	UFUNCTION(BlueprintCallable, Category="DuckiebotLoopStatus")
	bool UpdateRecentlyVisitedRoadTiles(FString NewRoadTileName);

private:
	float CommandArray[8];

	TCircularQueue<FString> RecentlyVisitedRoadTilesQueue = TCircularQueue<FString>(5);
	TSet<FString> RecentlyVisitedRoadTilesSet;
};
