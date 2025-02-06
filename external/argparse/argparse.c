/*
Copyright (C) 2012-2015 Yecheng Fu <cofyc.jackson at gmail dot com>
All rights reserved.
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdbool.h>
#include <stdarg.h>
#include <errno.h> // NV: added errno.h, it was missing

#include "argparse.h"

#define OPT_UNSET 1
#define OPT_LONG (1 << 1)

static const char*
prefix_skip(const char* str, const char* prefix) {
    size_t len = strlen(prefix);

    return strncmp(str, prefix, len) ? NULL : str + len;
}

static int
prefix_cmp(const char* str, const char* prefix) {
    for (;; str++, prefix++)
        if (!*prefix)
            return 0;
        else if (*str != *prefix)
            return (unsigned char)*prefix - (unsigned char)*str;
}

static int
argparse_printf(struct argparse* self, bool error, const char* fmt, ...)
{
    if (self->flags & ARGPARSE_USE_MESSAGE_BUFFER)
    {
        const size_t chunk_size = 4096;
        // Initial allocation of one chunk
        if (!self->messages)
        {
            self->message_buffer_size = chunk_size;
            self->messages = malloc(self->message_buffer_size);
        }

        // Try printing this message into the buffer at current offset, see if it fits.
        va_list args;
        va_start(args, fmt);
        size_t buffer_available = self->message_buffer_size - self->message_write_offset;
        int output_size = vsnprintf(self->messages + self->message_write_offset, buffer_available, fmt, args);
        
        // If the message doesn't fit, grow the buffer.
        if (output_size + 1 > (int)buffer_available)
        {
            // Grow the buffer size and round up to chunk size.
            self->message_buffer_size += (output_size + 1) - buffer_available;
            self->message_buffer_size = ((self->message_buffer_size + chunk_size - 1) / chunk_size) * chunk_size;
            self->messages = realloc(self->messages, self->message_buffer_size);

            // New buffer size, print again. It should succeed.
            buffer_available = self->message_buffer_size - self->message_write_offset;
            va_start(args, fmt);
            output_size = vsnprintf(self->messages + self->message_write_offset, buffer_available, fmt, args);
            assert(output_size + 1 <= (int)buffer_available);
        }

        // If the vsnprintf call was successful, advance the write pointer.
        if (output_size > 0)
            self->message_write_offset += output_size;
        return output_size;
    }
    else // !ARGPARSE_USE_MESSAGE_BUFFER
    {
        va_list args;
        va_start(args, fmt);
        return vfprintf(error ? stderr : stdout, fmt, args);
    }
}

static int
argparse_error(struct argparse* self, const struct argparse_option* opt, const char* reason, int flags) {
    if (flags & OPT_LONG)
        argparse_printf(self, true, "error: option `--%s` %s\n", opt->long_name, reason);
    else
        argparse_printf(self, true, "error: option `-%c` %s\n", opt->short_name, reason);

    if (!(self->flags & ARGPARSE_NEVER_EXIT))
        exit(EXIT_FAILURE);
    return ARGPARSE_INVALID_VALUE;
}

static int
argparse_getvalue(struct argparse* self, const struct argparse_option* opt, int flags) {
    const char* s = NULL;
    if (!opt->value)
        goto skipped;

    switch (opt->type) {
        case ARGPARSE_OPT_BOOLEAN:
            // NV: replaced the original logic that was casting these flags to 'int' and testing their sign
            // with this version that casts them to 'bool' to make it compatible with bool variables in C++.
            if (flags & OPT_UNSET)
                *(bool*)opt->value = false;
            else
                *(bool*)opt->value = true;
            break;
        case ARGPARSE_OPT_BIT:
            if (flags & OPT_UNSET)
                *(int*)opt->value &= ~opt->data;
            else
                *(int*)opt->value |= opt->data;
            break;
        case ARGPARSE_OPT_STRING:
            if (self->optvalue) {
                *(const char**)opt->value = self->optvalue;
                self->optvalue = NULL;
            } else if (self->argc > 1) {
                self->argc--;
                *(const char**)opt->value = *++self->argv;
            } else {
                return argparse_error(self, opt, "requires a value", flags);
            }
            break;
        case ARGPARSE_OPT_INTEGER:
            errno = 0;
            if (self->optvalue) {
                *(int*)opt->value = strtol(self->optvalue, (char**)&s, 0);
                self->optvalue = NULL;
            } else if (self->argc > 1) {
                self->argc--;
                *(int*)opt->value = strtol(*++self->argv, (char**)&s, 0);
            } else {
                return argparse_error(self, opt, "requires a value", flags);
            }
            if (errno == ERANGE)
                return argparse_error(self, opt, "numerical result out of range", flags);
            if (s[0] != '\0') // no digits or contains invalid characters
                return argparse_error(self, opt, "expects an integer value", flags);
            break;
        case ARGPARSE_OPT_FLOAT:
            errno = 0;
            if (self->optvalue) {
                *(float*)opt->value = strtof(self->optvalue, (char**)&s);
                self->optvalue = NULL;
            } else if (self->argc > 1) {
                self->argc--;
                *(float*)opt->value = strtof(*++self->argv, (char**)&s);
            } else {
                return argparse_error(self, opt, "requires a value", flags);
            }
            if (errno == ERANGE)
                return argparse_error(self, opt, "numerical result out of range", flags);
            if (s[0] != '\0') // no digits or contains invalid characters
                return argparse_error(self, opt, "expects a numerical value", flags);
            break;
        default:
            assert(0);
    }

skipped:
    if (opt->callback)
        return opt->callback(self, opt);

    return 0;
}

static void
argparse_options_check(struct argparse* self, const struct argparse_option* options) {
    for (; options->type != ARGPARSE_OPT_END; options++) {
        switch (options->type) {
            case ARGPARSE_OPT_END:
            case ARGPARSE_OPT_BOOLEAN:
            case ARGPARSE_OPT_BIT:
            case ARGPARSE_OPT_INTEGER:
            case ARGPARSE_OPT_FLOAT:
            case ARGPARSE_OPT_STRING:
            case ARGPARSE_OPT_GROUP:
                continue;
            default:
                argparse_printf(self, true, "wrong option type: %d", options->type);
                break;
        }
    }
}

static int
argparse_short_opt(struct argparse* self, const struct argparse_option* options) {
    for (; options->type != ARGPARSE_OPT_END; options++) {
        if (options->short_name == *self->optvalue) {
            self->optvalue = self->optvalue[1] ? self->optvalue + 1 : NULL;
            return argparse_getvalue(self, options, 0);
        }
    }

    return ARGPARSE_UNKNOWN_ARGUMENT;
}

static int
argparse_long_opt(struct argparse* self, const struct argparse_option* options) {
    for (; options->type != ARGPARSE_OPT_END; options++) {
        const char* rest;
        int opt_flags = 0;
        if (!options->long_name)
            continue;

        rest = prefix_skip(self->argv[0] + 2, options->long_name);
        if (!rest) {
            // negation disabled?
            if (options->flags & OPT_NONEG)
                continue;
            // only OPT_BOOLEAN/OPT_BIT supports negation
            if (options->type != ARGPARSE_OPT_BOOLEAN && options->type != ARGPARSE_OPT_BIT)
                continue;

            if (prefix_cmp(self->argv[0] + 2, "no-"))
                continue;
            rest = prefix_skip(self->argv[0] + 2 + 3, options->long_name);
            if (!rest)
                continue;
            opt_flags |= OPT_UNSET;
        }
        if (*rest) {
            if (*rest != '=')
                continue;
            self->optvalue = rest + 1;
        }
        return argparse_getvalue(self, options, opt_flags | OPT_LONG);
    }
    return ARGPARSE_UNKNOWN_ARGUMENT;
}

int argparse_init(struct argparse* self, struct argparse_option* options, const char* const* usages, int flags) {
    memset(self, 0, sizeof(*self));
    self->options = options;
    self->usages = usages;
    self->flags = flags;
    self->description = NULL;
    self->epilog = NULL;
    self->messages = NULL;
    self->message_write_offset = 0;
    self->message_buffer_size = 0;

    return 0;
}

void argparse_cleanup(struct argparse *self) {
    free(self->messages);
    self->messages = NULL;
    self->message_buffer_size = 0;
    self->message_write_offset = 0;
}

void argparse_describe(struct argparse* self, const char* description, const char* epilog) {
    self->description = description;
    self->epilog = epilog;
}

int argparse_parse(struct argparse* self, int argc, const char** argv) {
    self->argc = argc - 1;
    self->argv = argv + 1;
    self->out = argv;

    argparse_options_check(self, self->options);

    for (; self->argc; self->argc--, self->argv++) {
        const char* arg = self->argv[0];
        if (arg[0] != '-' || !arg[1]) {
            if (self->flags & ARGPARSE_STOP_AT_NON_OPTION)
                goto end;
            // if it's not option or is a single char '-', copy verbatim
            self->out[self->cpidx++] = self->argv[0];
            continue;
        }
        // short option
        if (arg[1] != '-') {
            self->optvalue = arg + 1;
            while (self->optvalue) {
                int opt_result = argparse_short_opt(self, self->options);
                if (opt_result == ARGPARSE_UNKNOWN_ARGUMENT)
                    goto unknown;
                else if (opt_result < 0)
                    return opt_result;
            }
            continue;
        }
        // if '--' presents
        if (!arg[2]) {
            self->argc--;
            self->argv++;
            break;
        }
        // long option
        int opt_result = argparse_long_opt(self, self->options);
        if (opt_result == ARGPARSE_UNKNOWN_ARGUMENT)
            goto unknown;
        else if (opt_result < 0)
            return opt_result;
        continue;

    unknown:
        argparse_printf(self, true, "error: unknown option `%s`\n\n", self->argv[0]);
        argparse_usage(self);
        if (!(self->flags & ARGPARSE_IGNORE_UNKNOWN_ARGS))
        {
            if (self->flags & ARGPARSE_NEVER_EXIT)
                return ARGPARSE_UNKNOWN_ARGUMENT;
            else
                exit(EXIT_FAILURE);
        }
    }

end:
    // NV: added (char**) cast to remove the const qualifier that C++ complains about
    memmove((char**)(self->out + self->cpidx), self->argv, self->argc * sizeof(*self->out));
    self->out[self->cpidx + self->argc] = NULL;

    return self->cpidx + self->argc;
}

void argparse_usage(struct argparse* self) {
    if (self->usages) {
        argparse_printf(self, false, "Usage: %s\n", *self->usages++);
        while (*self->usages && **self->usages)
            argparse_printf(self, false, "   or: %s\n", *self->usages++);
    } else {
        argparse_printf(self, false, "Usage:\n");
    }

    // print description
    if (self->description)
        argparse_printf(self, false, "%s\n", self->description);

    argparse_printf(self, false, "\n");

    const struct argparse_option* options;

    // figure out best width
    size_t usage_opts_width = 0;
    size_t len;
    options = self->options;
    for (; options->type != ARGPARSE_OPT_END; options++) {
        len = 0;
        if ((options)->short_name)
            len += 2;
        if ((options)->short_name && (options)->long_name)
            len += 2; // separator ", "

        if ((options)->long_name)
            len += strlen((options)->long_name) + 2;
        if (options->type == ARGPARSE_OPT_INTEGER)
            len += strlen("=<int>");
        if (options->type == ARGPARSE_OPT_FLOAT)
            len += strlen("=<flt>");
        else if (options->type == ARGPARSE_OPT_STRING)
            len += strlen("=<str>");
        len = (len + 3) - ((len + 3) & 3);
        if (usage_opts_width < len)
            usage_opts_width = len;
    }
    usage_opts_width += 4; // 4 spaces prefix

    options = self->options;
    for (; options->type != ARGPARSE_OPT_END; options++) {
        size_t pos = 0;
        size_t pad = 0;
        if (options->type == ARGPARSE_OPT_GROUP) {
            argparse_printf(self, false, "\n%s\n", options->help);
            continue;
        }
        pos = argparse_printf(self, false, "    ");
        if (options->short_name)
            pos += argparse_printf(self, false, "-%c", options->short_name);
        if (options->long_name && options->short_name)
            pos += argparse_printf(self, false, ", ");
        if (options->long_name)
            pos += argparse_printf(self, false, "--%s", options->long_name);
        if (options->type == ARGPARSE_OPT_INTEGER)
            pos += argparse_printf(self, false, "=<int>");
        else if (options->type == ARGPARSE_OPT_FLOAT)
            pos += argparse_printf(self, false, "=<flt>");
        else if (options->type == ARGPARSE_OPT_STRING)
            pos += argparse_printf(self, false, "=<str>");
        if (pos <= usage_opts_width) {
            pad = usage_opts_width - pos;
        } else {
            argparse_printf(self, false, "\n");
            pad = usage_opts_width;
        }
        argparse_printf(self, false, "%*s%s\n", (int)pad + 2, "", options->help);
    }

    // print epilog
    if (self->epilog)
        argparse_printf(self, false, "%s\n", self->epilog);
}

int argparse_help_cb_no_exit(struct argparse* self, const struct argparse_option* option) {
    (void)option;
    argparse_usage(self);

    return (EXIT_SUCCESS);
}

int argparse_help_cb(struct argparse* self, const struct argparse_option* option) {
    argparse_help_cb_no_exit(self, option);
    if (self->flags & ARGPARSE_NEVER_EXIT)
        return ARGPARSE_HELP;
    else
        exit(EXIT_SUCCESS);
}
