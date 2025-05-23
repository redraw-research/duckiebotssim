// Fill out your copyright notice in the Description page of Project Settings.


#include "DuckiebotAgent.h"

#include "DuckiebotPawnMovement.h"


// Sets default values
ADuckiebotAgent::ADuckiebotAgent()
{
	// Set this pawn to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

	// Set the defualt controller
	AIControllerClass = LoadClass<AController>(NULL, TEXT("/Script/DuckiebotsSim.DuckiebotAgentController"), NULL, LOAD_None, NULL);
	AutoPossessAI = EAutoPossessAI::PlacedInWorld;
}

void ADuckiebotAgent::InitializeAgent() {
	Super::InitializeAgent();
}

// Called every frame
void ADuckiebotAgent::Tick(float DeltaTime)
{
	// UE_LOG(LogDuckiebotsSim, Warning, TEXT("Duckiebot Agent Tick %d"), GFrameNumber);
	Super::Tick(DeltaTime);
}

float ADuckiebotAgent::GetVelocityCommand()
{
	return FMath::Clamp(CommandArray[0], -MAX_DUCKIEBOT_FORWARD_SPEED, MAX_DUCKIEBOT_FORWARD_SPEED);
}

float ADuckiebotAgent::GetRotationCommand()
{
	return FMath::Clamp(CommandArray[1], -MAX_DUCKIEBOT_ROTATION_SPEED, MAX_DUCKIEBOT_ROTATION_SPEED);
}

bool ADuckiebotAgent::GetResetCommand()
{
	return CommandArray[2] > 0;
}

void ADuckiebotAgent::ConsumeResetCommand()
{
	CommandArray[2] = 0.f;
}

FVector3f ADuckiebotAgent::GetResetXYYawCommand()
{
	return FVector3f(CommandArray[3], CommandArray[4], CommandArray[5]);
}

float ADuckiebotAgent::GetResetForwardVelCommand()
{
	return CommandArray[6];
}

float ADuckiebotAgent::GetResetYawVelCommand()
{
	return CommandArray[7];
}

void ADuckiebotAgent::ClearRewardTrackingState()
{
	RecentlyVisitedRoadTilesQueue.Empty();
	RecentlyVisitedRoadTilesSet.Empty();
}

bool ADuckiebotAgent::UpdateRecentlyVisitedRoadTiles(FString NewRoadTileName)
{
	// returns true if the NewRoadTileName is not recently visited 
	if (RecentlyVisitedRoadTilesSet.Contains(NewRoadTileName))
	{
		return false;
	}
	
	RecentlyVisitedRoadTilesSet.Add(NewRoadTileName);
	if (RecentlyVisitedRoadTilesQueue.IsFull())
	{
		// Remove oldest road tile that's about to be ejected from the queue.
		const FString* OldestRoadTile = RecentlyVisitedRoadTilesQueue.Peek();
		RecentlyVisitedRoadTilesSet.Remove(*OldestRoadTile);
	}
	RecentlyVisitedRoadTilesQueue.Enqueue(NewRoadTileName);
	return true;
}



