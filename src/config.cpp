#include "TaskArgs.h"

#include <vector_types.h>
#include <QList>
#include <QSize>
#include <QString>

namespace ProjConf
{
// clang-format off

extern constexpr QSize DEFAULT_MAINWINDOW_SIZE { 1024, 768 };

// <dGrid, dBlock>
extern QList<std::pair<dim3, dim3>> DIMENSION_OPTIONS {
    { dim3(10, 1080), dim3(192, 1) },   // 1920 x 1080
    { dim3(10, 1440), dim3(256, 1) },   // 2560 x 1440
    { dim3(15, 2160), dim3(256, 1) },   // 3840 x 2160
};

extern QString const DEFAULT_CENTER_X {QStringLiteral("-0.7438")};
extern QString const DEFAULT_CENTER_Y {QStringLiteral("0.148")};
extern QString const DEFAULT_HALF_UNIT {QStringLiteral("0.000002")};
extern QString const DEFAULT_ITER_LIMIT {QStringLiteral("4000")};

extern constexpr int COLOR_RANGE_SIXTH = 1048576; // pow(2, 20)

extern QString const INPUT_PANE_STYLE {QStringLiteral(R"(
    * {
        font-size: 16px;
    }

    QScrollArea {
        min-width: 240px;
        background: transparent;
        border: none;
    }

    QLineEdit {
        border: none;
    }
    
)")};

extern QString const TEXTURE_VIEW_STYLE {QStringLiteral(R"(
    min-width: 200px;
    min-height: 200px;
)")};

extern QString const CHANNEL_CURVES_STYLE{QStringLiteral(R"(
    min-width: 100px;
    min-height: 100px;
)")};

extern char const* SHADER_TEXTURE_VIEW_VERTEX =
#include "shaders/TextureViewVertex.str"
;

extern char const* SHADER_TEXTURE_VIEW_FRAGMENT =
#include "shaders/TextureViewFragment.str"
;

extern constexpr float CHANNEL_CURVES_STROKE_PEAK = 1.5;
extern constexpr float CHANNEL_CURVES_STROKE_NORM = 1.0 / 0.0125;

extern char const* SHADER_CHANNEL_CURVES_VERTEX =
#include "shaders/ChannelCurvesVertex.str"
;

extern char const* SHADER_CHANNEL_CURVES_FRAGMENT =
#include "shaders/ChannelCurvesFragment.str"
;

// clang-format on
}  // namespace ProjConf
