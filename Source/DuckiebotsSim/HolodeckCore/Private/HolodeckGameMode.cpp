// Fill out your copyright notice in the Description page of Project Settings.


#include "HolodeckGameMode.h"
#include "Kismet/GameplayStatics.h"
#include "DuckiebotsSim.h"

const char RESET_KEY[] = "RESET";
const int RESET_BYTES = 1;

AHolodeckGameMode::AHolodeckGameMode(const FObjectInitializer& ObjectInitializer) : AGameModeBase(ObjectInitializer), bHolodeckIsOn(true) {
	PrimaryActorTick.bCanEverTick = true;
	PrimaryActorTick.TickGroup = TG_PrePhysics;
	UE_LOG(LogDuckiebotsSim, Log, TEXT("HolodeckGameMode initialized"));
}

void AHolodeckGameMode::Tick(float DeltaSeconds) {
	Super::Tick(DeltaSeconds);

	// If !bHolodeckIsOn, then we never got instance or reset signal,
	// so we don't need to check bOn here.
	if (this->Instance)
		this->Instance->Tick(DeltaSeconds);
	if (this->CommandCenter)
		this->CommandCenter->Tick(DeltaSeconds);
	//Check if we should reset, and then reset the level. 
	if (ResetSignal != nullptr && *ResetSignal) {
		// UGameplayStatics::OpenLevel(this->Instance, FName(*GetWorld()->GetName()), false);
		this->ResetEpisode();
		*ResetSignal = false;
	}

}

void AHolodeckGameMode::StartPlay() {
	UE_LOG(LogDuckiebotsSim, Log, TEXT("HolodeckGameMode starting play"));

	// To prevent crashing in standalone games, check the HolodeckOn command is supplied.
	// This overrides the bHolodeckIsOn value supplied in the editor.
	if (GetWorld()->WorldType == EWorldType::Game)
		bHolodeckIsOn = FParse::Param(FCommandLine::Get(), TEXT("HolodeckOn"));
	UE_LOG(LogDuckiebotsSim, Warning, TEXT("bHolodeckIsOn: %d"), bHolodeckIsOn);
	
	if (bHolodeckIsOn) {
		this->Instance = (UHolodeckGameInstance*)(GetGameInstance());
		if (this->Instance) {
			this->Instance->StartServer();
			Server = this->Instance->GetServer();

			RegisterSettings();
		}
		else {
			UE_LOG(LogDuckiebotsSim, Warning, TEXT("Game Instance couldn't be found and initialized"));
		}
	if (this->Server) {
			this->CommandCenter = NewObject<UCommandCenter>();
			CommandCenter->Init(Server, this);
		}
	}

	Super::StartPlay();
}

void AHolodeckGameMode::RegisterSettings() {
	UE_LOG(LogDuckiebotsSim, Log, TEXT("Registering Settings"));
	if (Server != nullptr) {
		ResetSignal = static_cast<bool*>(Server->Malloc(RESET_KEY, RESET_BYTES));
		UE_LOG(LogDuckiebotsSim, Log, TEXT("Reset signal registered"));
	}
}

void AHolodeckGameMode::LogFatalMessage(const FString& Message) {
	UE_LOG(LogDuckiebotsSim, Fatal, TEXT("%s"), *Message);
}


