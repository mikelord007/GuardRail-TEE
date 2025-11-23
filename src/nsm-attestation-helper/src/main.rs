// Copyright (c), Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use nsm_api::api::{Request as NsmRequest, Response as NsmResponse};
use nsm_api::driver;
use serde_bytes::ByteBuf;
use std::env;

fn main() {
    // Read public key from command line argument (hex-encoded)
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <public_key_hex>", args[0]);
        eprintln!("Example: {} 24f56c2bab233ec5ccf61e50f641fee93f8aa3536d850e28b068370d43d57044", args[0]);
        std::process::exit(1);
    }

    let public_key_hex = &args[1];
    
    // Decode hex public key
    let public_key_bytes = match hex::decode(public_key_hex) {
        Ok(bytes) => bytes,
        Err(e) => {
            eprintln!("Error: Failed to decode public key hex: {}", e);
            std::process::exit(1);
        }
    };

    // Initialize NSM
    let fd = driver::nsm_init();

    // Create attestation request with public key
    let request = NsmRequest::Attestation {
        user_data: None,
        nonce: None,
        public_key: Some(ByteBuf::from(public_key_bytes)),
    };

    // Process request
    let response = driver::nsm_process_request(fd, request);
    
    // Handle response
    match response {
        NsmResponse::Attestation { document } => {
            driver::nsm_exit(fd);
            // Output hex-encoded attestation document to stdout
            let attestation_hex = hex::encode(document);
            println!("{}", attestation_hex);
            std::process::exit(0);
        }
        NsmResponse::Error(error) => {
            driver::nsm_exit(fd);
            eprintln!("Error: NSM returned error: {:?}", error);
            std::process::exit(1);
        }
        _ => {
            driver::nsm_exit(fd);
            eprintln!("Error: Unexpected NSM response type");
            std::process::exit(1);
        }
    }
}

