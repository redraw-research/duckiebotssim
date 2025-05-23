#pragma once

#include "DuckiebotsSim.h"

#include "HolodeckControlScheme.generated.h"

/**
  * UHolodeckControlScheme
  */
UCLASS()
class DUCKIEBOTSSIM_API UHolodeckControlScheme : public UObject {
	GENERATED_BODY()

public:
	UHolodeckControlScheme();
	UHolodeckControlScheme(const FObjectInitializer& ObjectInitializer);

	virtual void Execute(void* const CommandArray, void* const InputCommand, float DeltaSeconds);

	virtual unsigned int GetControlSchemeSizeInBytes() const;
};
