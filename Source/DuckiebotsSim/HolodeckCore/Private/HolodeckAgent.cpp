// MIT License (c) 2019 BYU PCCL see LICENSE file

#include "HolodeckAgent.h"
#include "DuckiebotsSim.h"
#include "HolodeckSensor.h"
#include "Components/PrimitiveComponent.h"

AHolodeckAgent::AHolodeckAgent() {
	PrimaryActorTick.bCanEverTick = true;
	PrimaryActorTick.TickGroup = TG_PrePhysics; //The tick function will we called before any physics simulation. 
	AddTickPrerequisiteActor(GetController()); //The agent's controller will always tick before the agent.
}

void AHolodeckAgent::BeginPlay(){
	Super::BeginPlay();
}

void AHolodeckAgent::InitializeAgent() {

	UE_LOG(LogDuckiebotsSim, Log, TEXT("Initializing HolodeckAgent %s"), *AgentName);
	if (!InitializeController())
		UE_LOG(LogDuckiebotsSim, Warning, TEXT("Couldn't initialize HolodeckPawnController for HolodeckAgent."));

	//Need to initialize this so that collision events will work (OnActorHit won't be called without it)
	//This is needed specifically for the collision sensor.
	if (UPrimitiveComponent* PrimitiveComponent = Cast<UPrimitiveComponent>(RootComponent)) {
		PrimitiveComponent->SetNotifyRigidBodyCollision(true);
		UE_LOG(LogDuckiebotsSim, Log, TEXT("HolodeckAgent collision events enabled"));
	} else {
		UE_LOG(LogDuckiebotsSim, Warning, TEXT("HolodeckAgent unable to get UPrimitiveComponent. Collision events disabled."));
	}

	Instance = static_cast<UHolodeckGameInstance*>(GetGameInstance());
	Server = Instance->GetServer();

	UE_LOG(LogDuckiebotsSim, Log, TEXT("Adding Agent %s to Server"), *AgentName);
	if (Server == nullptr) {
		UE_LOG(LogDuckiebotsSim, Warning, TEXT("Agent could not find server..."));
	} else {
		Server->AgentMap.Add(*AgentName, this);
	}

	// Initialize Sensors
	TArray<UHolodeckSensor*> Sensors;
	this->GetComponents(Sensors, true);
	
	for (auto& ActorSensor : Sensors) {
		UHolodeckSensor* Sensor = Cast<UHolodeckSensor>(ActorSensor);
		Sensor->SetAgentAndController(HolodeckController, AgentName);
		Sensor->InitializeSensor();
		this->SensorMap.Add(Sensor->SensorName, Sensor);
	}
}

void AHolodeckAgent::Tick(float DeltaSeconds) {
	Super::Tick(DeltaSeconds);
	MaxHeightCeiling();
}

bool AHolodeckAgent::Teleport(const FVector& NewLocation, const FRotator& NewRotation) {
	UE_LOG(LogDuckiebotsSim, Log, TEXT("Attempting to teleport HolodeckAgent %s"), *AgentName);
	FHitResult DummyHitResult;
	bool bWasSuccessful = this->K2_SetActorLocationAndRotation(
		NewLocation,
		NewRotation,
		false, //will not be blocked by object in between current and new location. 
		DummyHitResult, //this object is where the hit result is reported, if teleport can be blocked by objects in between.
		true //the object will retain its momentum(otherwise the android could not be teleported).
	);

	if (bWasSuccessful) {
		UE_LOG(LogDuckiebotsSim, Log, TEXT("HolodeckAgent %s teleported successfully"), *AgentName);
	} else {
		UE_LOG(LogDuckiebotsSim, Warning, TEXT("HolodeckAgent %s did not teleport successfully"), *AgentName);
	}
	
	return bWasSuccessful;
}

bool AHolodeckAgent::Teleport(const FVector& NewLocation){
	FRotator DefaultRotation = this->GetActorRotation();
	return Teleport(NewLocation, DefaultRotation);
}

bool AHolodeckAgent::SetState(const FVector& NewLocation, const FRotator& NewRotation, const FVector& NewVelocity, const FVector& NewAngVelocity) {

	FHitResult HitResult;
	bool bWasSuccessful = this->K2_SetActorLocationAndRotation(
		NewLocation,
		NewRotation,
		false, // don't sweep, otherwise will be blocked by an object in the path
		HitResult, //this object is where the hit result is reported, if teleport can be blocked by objects in between.
		false 
	);

	UPrimitiveComponent* RootComp = (UPrimitiveComponent*)this->GetRootComponent();

	FRotator RotationNow = RootComp->GetComponentRotation();
	FVector AngularVelocityVector = RotationNow.RotateVector(NewAngVelocity); //Rotate from local angles to world angles

	RootComp->SetAllPhysicsLinearVelocity(NewVelocity, false);
	RootComp->SetAllPhysicsAngularVelocityInDegrees(AngularVelocityVector, false);

	return bWasSuccessful;
}

bool AHolodeckAgent::InitializeController() {
	UE_LOG(LogDuckiebotsSim, Log, TEXT("Attempting to initialize controller for HolodeckAgent"));

	if (AgentName == "") {
		UE_LOG(LogDuckiebotsSim, Error, TEXT("HolodeckAgent doesn't have a name! Must set HolodeckAgent's name"));
		return false;
	}

	HolodeckController = static_cast<AHolodeckPawnController*>(Controller);

	if (HolodeckController == nullptr) {
		UE_LOG(LogDuckiebotsSim, Error, TEXT("Couldn't find a HolodeckPawnController for HolodeckAgent"));
		return false;
	} else {
		// We found the controller, so tell it to set up the action buffers.
		HolodeckController->AllocateBuffers(AgentName);
		UE_LOG(LogDuckiebotsSim, Log, TEXT("HolodeckAgent controller setup was successful"));
		return true;
	}
}

void AHolodeckAgent::MaxHeightCeiling() {
	FVector origin, extent;
	GetActorBounds(true, origin, extent);
	double actorHeight = extent.Z * 2;
	double actorMaxHeight = MaxHeight - actorHeight;

	if (GetActorLocation().Z >= actorMaxHeight) {
		FVector Position(
			GetActorLocation().X,
			GetActorLocation().Y,
			actorMaxHeight
		);
		Teleport(Position);
	}
}
