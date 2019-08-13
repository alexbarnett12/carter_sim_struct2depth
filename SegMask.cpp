#include "SegMask.hpp"
#include <cstdio>


void SegMask::start()
{
    tickOnMessage(rx_seg_masks());
}
void SegMask::tick()
{
    LOG_INFO("ping");
}
void SegMask::stop() {}