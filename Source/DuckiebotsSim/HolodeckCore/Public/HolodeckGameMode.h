// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "DuckiebotsSim.h"

#include "CoreMinimal.h"
#include "GameFramework/GameModeBase.h"
#include "HolodeckGameInstance.h"
#include "HolodeckServer.h"
#include "CommandCenter.h"
#include "HolodeckGameMode.generated.h"


/**
 * AHolodeckGameMode
 * The base game mode for Holodeck.
 * HolodeckGameModeBP can be used to turn on and off Holodeck functionality
 * from the editor.
 */
UCLASS()
class DUCKIEBOTSSIM_API AHolodeckGameMode : public AGameModeBase
{
	GENERATED_BODY()
public:
	/**
	  * Default Constructor.
	  */
	explicit AHolodeckGameMode(const FObjectInitializer& ObjectInitializer);

	/**
	  * Tick
	  * Ticks the game mode.
	  * Checks the change in settings, and ticks the game instance.
	  * @param DeltaSeconds how many seconds passed since last tick.
	  */
	void Tick(float DeltaSeconds) override;

	/**
	  * StartPlay
	  * Called when the game begins.
	  * Registers all settings.
	  */
	void StartPlay() override;

	// Can be set off to turn off holodeck functionality.
	UPROPERTY(EditAnywhere)
		bool bHolodeckIsOn;

	/**
	  * GetAssociatedServer
	  * Returns the private server pointer that the instance contains.
	  */
	UHolodeckServer* GetAssociatedServer() { return this->Server; };

	 // These functions allow the Holodeck to do things which cannot normally be done from pure c++ code
	UFUNCTION(BlueprintImplementableEvent)
		AHolodeckAgent* SpawnAgent(const FString& Type, const FVector& Location, const FRotator& Rotation, const FString& Name, bool IsMainAgent);

	UFUNCTION(BlueprintImplementableEvent)
		void TeleportCamera(const FVector& Location, const FVector& Rotation);

	UFUNCTION(BlueprintImplementableEvent)
		void ExecuteCustomCommand(const FString& Name, const TArray<float>& NumberParameters, const TArray<FString>& StringParameters);

	UFUNCTION(BlueprintImplementableEvent)
		AActor* FindActorWithTag(const FString& Tag);

	UFUNCTION(BlueprintImplementableEvent)
		float GetWorldNum(const FString& Key);

	UFUNCTION(BlueprintImplementableEvent)
		FString GetWorldString(const FString& Key);

	UFUNCTION(BlueprintImplementableEvent)
		bool GetWorldBool(const FString& Key);

	UFUNCTION(BlueprintImplementableEvent)
		void ResetEpisode();
	
	UFUNCTION(BlueprintCallable)
		void LogFatalMessage(const FString& Message);

private:
	/**
	  * RegisterSettings
	  * Registers all the settings on the server.
	  */
	void RegisterSettings();

	// Setting buffers
	bool* ResetSignal;

	UPROPERTY()
		UHolodeckServer* Server;
	
	UPROPERTY()
		UHolodeckGameInstance* Instance;
	
	UPROPERTY()
		UCommandCenter* CommandCenter;
};
