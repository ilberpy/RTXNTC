/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "Profiler.h"
#include <imgui.h>
#include <implot.h>

void AveragingTimerQuery::beginQuery(nvrhi::ICommandList* commandList)
{
    nvrhi::TimerQueryHandle query;
    if (m_idleQueries.empty())
    {
        query = m_device->createTimerQuery();
    }
    else
    {
        query = m_idleQueries.front();
        m_idleQueries.pop();
    }

    commandList->beginTimerQuery(query);
    m_openQuery = query;
}

void AveragingTimerQuery::endQuery(nvrhi::ICommandList* commandList)
{
    assert(m_openQuery);
    commandList->endTimerQuery(m_openQuery);
    m_activeQueries.push(m_openQuery);
    m_openQuery = nullptr;
}

void AveragingTimerQuery::update()
{
    while (!m_activeQueries.empty())
    {
        nvrhi::TimerQueryHandle query = m_activeQueries.front();
        if (m_device->pollTimerQuery(query))
        {
            float time = m_device->getTimerQueryTime(query);
            m_history.push_back(time);
            m_activeQueries.pop();
            m_idleQueries.push(query);
        }
        else
            break;
    }

    std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    float const secondsSinceUpdate = std::chrono::duration_cast<std::chrono::microseconds>(now - m_lastUpdateTime)
        .count() * 1e-6f;

    if (secondsSinceUpdate > m_updateIntervalSeconds && !m_history.empty())
    {
        float sumTime = 0.f;
        for (float time : m_history)
            sumTime += time;
        
        m_averageTime = sumTime / m_history.size();
        m_lastUpdateTime = now;

        float const latestTime = m_history[m_history.size() - 1];
        m_history.clear();
        m_history.push_back(latestTime);
    }
}

void AveragingTimerQuery::setUpdateInterval(float seconds)
{
    m_updateIntervalSeconds = seconds;
}

void AveragingTimerQuery::clearHistory()
{
    m_history.clear();
    m_lastUpdateTime = std::chrono::steady_clock::now();
}

std::optional<float> AveragingTimerQuery::getLatestAvailableTime()
{
    return m_history.empty() ? std::optional<float>() :  m_history.back();
}

std::optional<float> AveragingTimerQuery::getAverageTime()
{
    return m_averageTime;
}

void SmoothAxisLimit::Update(double newMaximum, double lastFrameTimeSeconds)
{
    if (newMaximum <= 0.0)
        return;

    // Round the new maximum up to the nearest power of two
    newMaximum = std::pow(2.0, std::ceil(std::log2(newMaximum)));

    if (m_maximum == newMaximum)
        return;

    if (m_maximum == 0.0)
    {
        m_maximum = newMaximum;
        return;
    }

    double const adaptationSpeed = 4.0;
    bool const adjustUp = m_maximum < newMaximum;
    double const factor = exp(adaptationSpeed * lastFrameTimeSeconds * (adjustUp ? 1.0 : -1.0));
    m_maximum *= factor;

    if (adjustUp)
        m_maximum = std::min(m_maximum, newMaximum);
    else
        m_maximum = std::max(m_maximum, newMaximum);
}

ProfilerRecord& Profiler::AddRecord()
{
    auto& record = m_profilerHistory.emplace_back();
    record.timestamp = std::chrono::duration<double>(std::chrono::steady_clock::now() - m_appStartTime).count();
    return record;
}

ProfilerRecord* Profiler::GetLastRecord()
{
    if (m_profilerHistory.empty())
        return nullptr;

    return &m_profilerHistory.back();
}

void Profiler::TrimHistory()
{
    if (m_profilerHistory.empty())
        return;
        
    double latestTimestamp = m_profilerHistory.back().timestamp;
    double historyCutoffTime = latestTimestamp - m_profilerHistoryDuration;

    // Find the first record that is within the history window
    size_t index;
    for (index = 0; index < m_profilerHistory.size(); index++)
    {
        if (m_profilerHistory[index].timestamp >= historyCutoffTime)
            break;
    }

    if (index > 0)
    {
        // Remove all records that are older than the history duration
        m_profilerHistory.erase(m_profilerHistory.begin(), m_profilerHistory.begin() + index);
    }
}

constexpr double c_SecondsToMs = 1e3;

// Getter functions for ImPlot
class ProfilerGetters
{
public:
    static ImPlotPoint GetFrameTime(int idx, void* userData)
    {
        Profiler* profiler = static_cast<Profiler*>(userData);
        return ImPlotPoint(GetTimeValue(idx, profiler), profiler->m_profilerHistory[idx].frameTime * c_SecondsToMs);
    }

    static ImPlotPoint GetRenderTime(int idx, void* userData)
    {
        Profiler* profiler = static_cast<Profiler*>(userData);
        return ImPlotPoint(GetTimeValue(idx, profiler), profiler->m_profilerHistory[idx].renderTime * c_SecondsToMs);
    }

    static ImPlotPoint GetTranscodingTime(int idx, void* userData)
    {
        Profiler* profiler = static_cast<Profiler*>(userData);
        return ImPlotPoint(GetTimeValue(idx, profiler), profiler->m_profilerHistory[idx].transcodingTime * c_SecondsToMs);
    }

    static ImPlotPoint GetTilesAllocated(int idx, void* userData)
    {
        Profiler* profiler = static_cast<Profiler*>(userData);
        return ImPlotPoint(GetTimeValue(idx, profiler), double(profiler->m_profilerHistory[idx].tilesAllocated));
    }

    static ImPlotPoint GetTilesStandby(int idx, void* userData)
    {
        Profiler* profiler = static_cast<Profiler*>(userData);
        return ImPlotPoint(GetTimeValue(idx, profiler), double(profiler->m_profilerHistory[idx].tilesStandby));
    }

private:
    static double GetTimeValue(int idx, Profiler* profiler)
    {
        return profiler->m_profilerHistory[idx].timestamp - profiler->m_profilerHistory.back().timestamp;
    }
};

void Profiler::BuildUI(bool enableFeedbackStats)
{
    float const fontSize = ImGui::GetFontSize();

    char durationLabel[64];
    snprintf(durationLabel, sizeof durationLabel, "%.1f s", m_profilerHistoryDuration);
    ImGui::PushItemWidth(fontSize * 6.f);
    if (ImGui::BeginCombo("Plot Duration", durationLabel))
    {
        double durations[] = { 0.5, 1.0, 2.0, 5.0 };
        for (double duration : durations)
        {
            snprintf(durationLabel, sizeof durationLabel, "%.1f s", duration);
            if (ImGui::Selectable(durationLabel, m_profilerHistoryDuration == duration))
            {
                m_profilerHistoryDuration = duration;
            }
        }
        ImGui::EndCombo();
    }
    ImGui::PopItemWidth();

    if (m_profilerHistory.size() < 10)
        return;
        
    double const secondToMs = 1e3;
    size_t const historySize = m_profilerHistory.size();
    double maxTime = 0;
    double maxTiles = 0;

    for (auto const& record : m_profilerHistory)
    {
        maxTime = std::max(maxTime, record.frameTime);
        maxTime = std::max(maxTime, record.renderTime);
        if (enableFeedbackStats)
            maxTime = std::max(maxTime, record.transcodingTime);
            
        maxTiles = std::max(maxTiles, double(record.tilesAllocated));
        maxTiles = std::max(maxTiles, double(record.tilesStandby));
    }

    maxTime *= secondToMs;
    m_timePlotLimit.Update(maxTime, m_profilerHistory.back().frameTime);
    m_tilesPlotLimit.Update(maxTiles, m_profilerHistory.back().frameTime);
    
    if (ImPlot::BeginPlot("Frame Time", ImVec2(20.f * fontSize, 15.f * fontSize),
        ImPlotFlags_NoTitle | ImPlotFlags_NoMenus | ImPlotFlags_NoInputs))
    {
        ImPlot::SetupAxes("Time (s)", "Time (ms)");
        ImPlot::SetupAxesLimits(-m_profilerHistoryDuration, 0, 0, m_timePlotLimit.GetMaximum(), ImGuiCond_Always);
        ImPlot::PlotLineG("Frame Time", &ProfilerGetters::GetFrameTime, this, historySize);
        ImPlot::PlotLineG("Render Time", &ProfilerGetters::GetRenderTime, this, historySize);
        if (enableFeedbackStats)
            ImPlot::PlotLineG("Transcoding Time", &ProfilerGetters::GetTranscodingTime, this, historySize);
        ImPlot::EndPlot();
    }

    if (enableFeedbackStats &&
        ImPlot::BeginPlot("Texture Tiles", ImVec2(20.f * fontSize, 15.f * fontSize),
        ImPlotFlags_NoTitle | ImPlotFlags_NoMenus | ImPlotFlags_NoInputs))
    {
        ImPlot::SetupAxes("Time (s)", "Tiles");
        ImPlot::SetupAxesLimits(-m_profilerHistoryDuration, 0, 0, m_tilesPlotLimit.GetMaximum(), ImGuiCond_Always);
        ImPlot::PlotLineG("Tiles Allocated", &ProfilerGetters::GetTilesAllocated, this, historySize);
        ImPlot::PlotLineG("Tiles Standby", &ProfilerGetters::GetTilesStandby, this, historySize);
        ImPlot::EndPlot();
    }
}
