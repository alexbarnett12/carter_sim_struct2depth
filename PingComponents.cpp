#include "PingComponents.hpp"
void Ping::start() {
    tickPeriodically();
}
void Ping::tick() {
    LOG_INFO("ping");
}
void Ping::stop() {}