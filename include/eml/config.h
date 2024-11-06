#ifndef EML_CONFIG_H
#define EML_CONFIG_H

#include <eml/platform.h>

// Controls which word size to memory to
// Note: this is only used for internal temporary allocations
#define EML_CONFIG_ALIGNMENT 4

// Quantized datatypes
namespace eml
{

using qint8_t = int8_t;
using qint16_t = int16_t;
using qint32_t = int32_t;

} // namespace eml


#if !defined( NDEBUG )
#define EML_DEBUG
#endif

#ifndef EML_DEBUG
#define EML_ASSERT( expr, msg ) ( (void)0 )
#else
#define EML_ASSERT( expr, msg ) ( ( expr ) ? (void)0 : PlatformAssert( #expr, msg, __FILE__, __LINE__ ) )
#endif

#define EML_LOG_INFO( msg ) PlatformLog( "[INFO]: %s:%d\n\t%s", __FILE__, __LINE__, msg )

#define EML_VERSION "0.0.1"



#endif // EML_CONFIG_H