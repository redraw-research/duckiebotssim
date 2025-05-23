// Fill out your copyright notice in the Description page of Project Settings.


#include "DuckiebotPawnMovement.h"

#include "ComponentUtils.h"
#include "DuckiebotAgent.h"
#include "GameFramework/Pawn.h"
#include "GameFramework/Controller.h"
#include "GameFramework/WorldSettings.h"
#include "Components/PrimitiveComponent.h"
#include "Containers/Ticker.h"


UDuckiebotPawnMovement::UDuckiebotPawnMovement(const FObjectInitializer& ObjectInitializer)
	: Super(ObjectInitializer)
{
	bUseSimplePhysics = false;
	MaxSpeed = 1200.f;
	ReverseSpeedCoefficient= 0.6;
	Acceleration = 4000.f;
	Deceleration = 8000.f;
	TurningBoost = 8.0f;
	TurningDeadzone = 0.9f;
	TurningScale = 160.0f;
	VelDeadzone = 0.5f;                                                             
	VelScale = 0.55f;
	bPositionCorrected = false;

	ResetMoveState();
}

void UDuckiebotPawnMovement::SetYawInput(float Val)
{
	YawInput = Val;
}                                            

void UDuckiebotPawnMovement::SetVelInput(float Val)
{
	VelInput = Val;
}

float UDuckiebotPawnMovement::GetCurrentYawVelocityFromInputs()
{
	return YawVelocityFromInputs;
}

float UDuckiebotPawnMovement::GetCurrentForwardVelocity()
{
	const FQuat Rotation = UpdatedComponent->GetComponentQuat();
	FVector LocalFrameVelocity = Rotation.UnrotateVector(Velocity);
	if (FMath::Abs(LocalFrameVelocity.Y) > 1e-3 || FMath::Abs(LocalFrameVelocity.Z) > 1e-3)
	{
		// UE_LOG(LogDuckiebotsSim, Warning, TEXT("Local Velocity has non-zero compoenents in the Y or Z axis: %s"), *LocalFrameVelocity.ToString());
	}
	return LocalFrameVelocity.X;
}


void UDuckiebotPawnMovement::TickComponent(float DeltaTime, enum ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	
	if (ShouldSkipUpdate(DeltaTime))
	{
		return;
	}

	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	if (!PawnOwner || !UpdatedComponent)
	{
		return;
	}
	
	TObjectPtr<class ADuckiebotAgent> DuckiebotAgentOwner = Cast<ADuckiebotAgent>(PawnOwner);
	if (nullptr != DuckiebotAgentOwner)
	{
		// UE_LOG(LogDuckiebotsSim, Warning, TEXT("Duckiebot MovementComponent Tick %d, Starting Velocity %s"), GFrameNumber, *UpdatedComponent->ComponentVelocity.ToString());
		// UE_LOG(LogDuckiebotsSim, Warning, TEXT("Duckiebot MovementComponent Tick %d, Starting Ang Velocity %s"), GFrameNumber, *UpdatedPrimitive->GetPhysicsAngularVelocityInDegrees().ToString());

		// FString ArrayString;
		// for (int32 i = 0; i < 8; ++i)
		// {
		// 	ArrayString += FString::Printf(TEXT("%f, "), DuckiebotAgentOwner->CommandArray[i]);
		// }
		// UE_LOG(LogDuckiebotsSim, Warning, TEXT("Tick %d, CommandArray: %s"), GFrameNumber, *ArrayString);


		bool DoReset = DuckiebotAgentOwner->GetResetCommand();
		if (DoReset)
		{
			FVector3f NewXYYaw = DuckiebotAgentOwner->GetResetXYYawCommand();
			NewXYYaw.X = ConvertClientDistanceToUnreal(NewXYYaw.X);
			NewXYYaw.Y = ConvertClientDistanceToUnreal(NewXYYaw.Y);
			// DuckiebotAgentOwner->ResetDuckiebotToNewLocation(NewXYYaw);
			FTransform OldTransform = UpdatedComponent->GetComponentTransform();
			FVector NewLocation(NewXYYaw.X, NewXYYaw.Y, OldTransform.GetLocation().Z);
			FRotator NewRotation(OldTransform.Rotator().Pitch, NewXYYaw.Z, OldTransform.Rotator().Roll);
			// UpdatedComponent->SetWorldLocationAndRotationNoPhysics(NewLocation, NewRotation);
			
			FVector LocalVelocity(DuckiebotAgentOwner->GetResetForwardVelCommand(), 0.0f, 0.0f);
			LocalVelocity = ConvertLinearVector(LocalVelocity, ClientToUE);
			FTransform LocalToWorldTransform = FTransform(NewRotation, NewLocation);
			// Convert the vector from local space to world space
			Velocity = LocalToWorldTransform.TransformVector(LocalVelocity);
			// UpdateComponentVelocity();
			// UE_LOG(LogDuckiebotsSim, Warning, TEXT("Tick %d: Velocity after original reset: %s"), GFrameNumber, *UpdatedComponent->ComponentVelocity.ToString());

			// Set Physics Angular Velocity
			float NewYalVel = DuckiebotAgentOwner->GetResetYawVelCommand();
			YawVelocityFromInputs = NewYalVel;
			FVector NewAngularVelocityVector(0.f, 0.f, NewYalVel);
			NewAngularVelocityVector = NewRotation.RotateVector(NewAngularVelocityVector);
			// UpdatedPrimitive->SetAllPhysicsAngularVelocityInDegrees(NewAngularVelocityVector);

			// Set Physics Linear Velocity
			// UpdatedPrimitive->SetAllPhysicsLinearVelocity(Velocity);

			DuckiebotAgentOwner->SetState(NewLocation, NewRotation, Velocity, NewAngularVelocityVector);
			
			// UE_LOG(LogDuckiebotsSim, Warning, TEXT("Tick %d, Ang Velocity after reset %s"), GFrameNumber, *UpdatedPrimitive->GetPhysicsAngularVelocityInDegrees().ToString());
			// UE_LOG(LogDuckiebotsSim, Warning, TEXT("Tick %d: Velocity after physics reset: %s"), GFrameNumber, *UpdatedComponent->ComponentVelocity.ToString());

			DuckiebotAgentOwner->ConsumeResetCommand();
			
		} else {
			SetVelInput(DuckiebotAgentOwner->GetVelocityCommand());
			SetYawInput(DuckiebotAgentOwner->GetRotationCommand());
			
			
			AController* Controller = PawnOwner->GetController();
			if (Controller && Controller->IsLocalController())
			{
				// apply input for local players but also for AI that's not following a navigation path at the moment
				if (Controller->IsLocalPlayerController() == true || Controller->IsFollowingAPath() == false || bUseAccelerationForPaths)
				{
					ApplyControlInputToVelocity(DeltaTime);
				}
				// if it's not player controller, but we do have a controller, then it's AI
				// (that's not following a path) and we need to limit the speed
				else if (IsExceedingMaxSpeed(MaxSpeed) == true)
				{
					Velocity = Velocity.GetUnsafeNormal() * MaxSpeed;
				}

				LimitWorldBounds();
				bPositionCorrected = false;

				// Move actor
				FVector Delta = Velocity * DeltaTime;

				float yawDelta = 0.f;
				if (bUseSimplePhysics) {
					yawDelta = YawInput * TurningScale * DeltaTime;
				} else {
					// Apply non-linear response curve to make handling more like real robot.
					float YawCurveCoef = 1.0f;
					float YawInputApplied = FMath::Sign(YawInput) * (FMath::Exp(FMath::Abs(YawInput) * YawCurveCoef) - 1.0) / (FMath::Exp(YawCurveCoef) - 1.0);
					// Apply turning dead-zone to mimic non-responsive robot handling when stationary or almost stationary.
					if (abs(YawInput) > TurningDeadzone || abs(VelInput) > VelDeadzone) {
						yawDelta = YawInputApplied * TurningScale * DeltaTime;
					} else {
						yawDelta = YawInputApplied * TurningScale * DeltaTime * (abs(YawInputApplied) / TurningDeadzone) * 0.01;
					}
				}
				
				YawVelocityFromInputs = yawDelta / DeltaTime;
					
				const FVector OldLocation = UpdatedComponent->GetComponentLocation();
				const FQuat Rotation = UpdatedComponent->GetComponentQuat();
					
				FRotator NewRotator(Rotation);
				NewRotator.Add(0, yawDelta, 0);
				Controller->SetControlRotation(NewRotator);
					
				FHitResult Hit(1.f);
				const FQuat NewRotation(NewRotator);
				SafeMoveUpdatedComponent(Delta, NewRotation, true, Hit);
					
					
				if (Hit.IsValidBlockingHit())
				{
					HandleImpact(Hit, DeltaTime, Delta);
					// Try to slide the remaining distance along the surface.
					SlideAlongSurface(Delta, 1.f - Hit.Time, Hit.Normal, Hit, true);
				}
					
				// Update velocity
				// We don't want position changes to vastly reverse our direction (which can happen due to penetration fixups etc)
				if (!bPositionCorrected)
				{
					const FVector NewLocation = UpdatedComponent->GetComponentLocation();
					Velocity = ((NewLocation - OldLocation) / DeltaTime);
				}
					
				// Finalize
				UpdateComponentVelocity();
				// UE_LOG(LogDuckiebotsSim, Warning, TEXT("Tick %d, Ending Velocity: %s"), GFrameNumber, *UpdatedComponent->ComponentVelocity.ToString());
				// UE_LOG(LogDuckiebotsSim, Warning, TEXT("Tick %d, End Ang Velocity: %s"), GFrameNumber, *UpdatedPrimitive->GetPhysicsAngularVelocityInDegrees().ToString());

				YawInput = 0.f;
			}
		}
	}
};

bool UDuckiebotPawnMovement::LimitWorldBounds()
{
	AWorldSettings* WorldSettings = PawnOwner ? PawnOwner->GetWorldSettings() : NULL;
	if (!WorldSettings || !WorldSettings->AreWorldBoundsChecksEnabled() || !UpdatedComponent)
	{
		return false;
	}

	const FVector CurrentLocation = UpdatedComponent->GetComponentLocation();
	if (CurrentLocation.Z < WorldSettings->KillZ)
	{
		Velocity.Z = FMath::Min<FVector::FReal>(GetMaxSpeed(), WorldSettings->KillZ - CurrentLocation.Z + 2.0f);
		return true;
	}

	return false;
}

void UDuckiebotPawnMovement::ApplyControlInputToVelocity(float DeltaTime)
{
	const FRotator Rotation = UpdatedComponent->GetComponentRotation();
	const FRotator YawRotation(0, Rotation.Yaw, 0);

	float VelInputApplied = 0.f;

	if (bUseSimplePhysics) {
		VelInputApplied = VelInput * VelScale;
	} else {
		if (abs(VelInput) > VelDeadzone) {
			float VelCurveCoef = 1.0f;
			VelInputApplied = FMath::Sign(VelInput) * (FMath::Exp(FMath::Abs(VelInput) * VelCurveCoef) - 1.0) / (FMath::Exp(VelCurveCoef) - 1.0);
			VelInputApplied = VelInputApplied * VelScale;
		}
	}
	if (VelInputApplied < 0.f)
	{
		VelInputApplied *= ReverseSpeedCoefficient;
	}

	const FVector ControlAcceleration = YawRotation.Vector() * VelInputApplied;


	//const FVector ControlAcceleration = GetPendingInputVector().GetClampedToMaxSize(1.f);

	const float AnalogInputModifier = (ControlAcceleration.SizeSquared() > 0.f ? ControlAcceleration.Size() : 0.f);
	const float MaxPawnSpeed = GetMaxSpeed() * AnalogInputModifier;
	const bool bExceedingMaxSpeed = IsExceedingMaxSpeed(MaxPawnSpeed);

	if (AnalogInputModifier > 0.f && !bExceedingMaxSpeed)
	{
		// Apply change in velocity direction
		if (Velocity.SizeSquared() > 0.f)
		{
			// Change direction faster than only using acceleration, but never increase velocity magnitude.
			const float TimeScale = FMath::Clamp(DeltaTime * TurningBoost, 0.f, 1.f);
			Velocity = Velocity + (ControlAcceleration * Velocity.Size() - Velocity) * TimeScale;
		}
	}
	else
	{
		// Dampen velocity magnitude based on deceleration.
		if (Velocity.SizeSquared() > 0.f)
		{
			const FVector OldVelocity = Velocity;
			const float VelSize = FMath::Max(Velocity.Size() - FMath::Abs(Deceleration) * DeltaTime, 0.f);
			Velocity = Velocity.GetSafeNormal() * VelSize;

			// Don't allow braking to lower us below max speed if we started above it.
			if (bExceedingMaxSpeed && Velocity.SizeSquared() < FMath::Square(MaxPawnSpeed))
			{
				Velocity = OldVelocity.GetSafeNormal() * MaxPawnSpeed;
			}
		}
	}

	// Apply acceleration and clamp velocity magnitude.
	const float NewMaxSpeed = (IsExceedingMaxSpeed(MaxPawnSpeed)) ? Velocity.Size() : MaxPawnSpeed;
	Velocity += ControlAcceleration * FMath::Abs(Acceleration) * DeltaTime;
	Velocity = Velocity.GetClampedToMaxSize(NewMaxSpeed);
	ConsumeInputVector();
}

bool UDuckiebotPawnMovement::ResolvePenetrationImpl(const FVector& Adjustment, const FHitResult& Hit, const FQuat& NewRotationQuat)
{
	bPositionCorrected |= Super::ResolvePenetrationImpl(Adjustment, Hit, NewRotationQuat);
	return bPositionCorrected;
}
