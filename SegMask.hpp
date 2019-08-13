#pragma once
#include "engine/alice/alice_codelet.hpp"
#include "messages/messages.hpp"


class SegMask : public isaac::alice::Codelet {
public:
    void start() override;

    void tick() override;

    void stop() override;

    ISAAC_PROTO_RX(SegmentationCameraProto, seg_masks
    )
};



ISAAC_ALICE_REGISTER_CODELET(SegMask);
