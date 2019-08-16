#include "SegMask.hpp"
#include <cstdio>

#include "engine/gems/image/color.hpp"
#include "engine/gems/image/io.hpp"
#include "engine/gems/image/utils.hpp"
#include "messages/camera.hpp"


void SegMask::start()
{
    tickOnMessage(rx_seg_masks());
}
void SegMask::tick()
{
//    auto proto = rx_seg_masks().getProto();
//    auto image = proto.getInstanceImage();

    isaac::ImageConstView3ub color_image;
    FromProto(rx_seg_masks().getProto().getInstanceImage(), rx_seg_masks().buffers(), color_image);
    SavePng(color_image, "path_to_file.png");


}
void SegMask::stop() {}